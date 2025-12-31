#!/usr/bin/env python3
"""
ZipVoice Inference Optimization Benchmark Script.

Tests various optimization approaches:
1. Different num_step values (16, 8, 4, 2)
2. Different models (zipvoice, zipvoice_distill)
3. Different backends (PyTorch, TensorRT if available)

Usage:
    uv run python scripts/benchmark_optimization.py \
        --prompt-wav data/prompt.wav \
        --prompt-text "Hello, this is a test." \
        --text "The quick brown fox jumps over the lazy dog."
"""

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)
from vocos import Vocos


HUGGINGFACE_REPO = "k2-fsa/ZipVoice"


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, EmiliaTokenizer, dict]:
    """Load model and tokenizer from HuggingFace."""
    model_dir = model_name

    model_ckpt = hf_hub_download(
        HUGGINGFACE_REPO, filename=f"{model_dir}/model.pt"
    )
    model_config_path = hf_hub_download(
        HUGGINGFACE_REPO, filename=f"{model_dir}/model.json"
    )
    token_file = hf_hub_download(
        HUGGINGFACE_REPO, filename=f"{model_dir}/tokens.txt"
    )

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    if model_name == "zipvoice":
        model = ZipVoice(**model_config["model"], **tokenizer_config)
    else:
        model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)

    load_checkpoint(filename=model_ckpt, model=model, strict=True)
    model = model.to(device).eval()

    return model, tokenizer, model_config


def run_single_inference(
    prompt_wav_path: str,
    prompt_text: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int,
    guidance_scale: float,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
) -> Dict[str, float]:
    """Run single inference with detailed timing."""
    timings = {}
    total_start = dt.datetime.now()

    # === Prompt Loading ===
    start = dt.datetime.now()
    prompt_wav = load_prompt_wav(prompt_wav_path, sampling_rate=sampling_rate)
    timings["prompt_load"] = (dt.datetime.now() - start).total_seconds()

    # === Prompt Preprocessing ===
    start = dt.datetime.now()
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / sampling_rate
    timings["prompt_preprocess"] = (dt.datetime.now() - start).total_seconds()

    # === Feature Extraction ===
    start = dt.datetime.now()
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    timings["feature_extraction"] = (dt.datetime.now() - start).total_seconds()

    # === Tokenization ===
    start = dt.datetime.now()
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    token_duration = prompt_duration / (len(prompt_tokens_str) * speed)
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    tokens_batches, chunked_index = batchify_tokens(
        chunked_tokens, 100, prompt_duration, token_duration
    )
    timings["tokenization"] = (dt.datetime.now() - start).total_seconds()

    # === Model Inference ===
    start = dt.datetime.now()
    chunked_features = []

    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)
        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full(
            (len(batch_tokens),), prompt_features.size(1), device=device
        )

        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        chunked_features.append((pred_features, pred_features_lens))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["model_inference"] = (dt.datetime.now() - start).total_seconds()

    # === Vocoder Decode ===
    start = dt.datetime.now()
    chunked_wavs = []

    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):
            wav = (
                vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
                .squeeze(1)
                .clamp(-1, 1)
            )
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["vocoder_decode"] = (dt.datetime.now() - start).total_seconds()

    # === Post Processing ===
    start = dt.datetime.now()
    indexed_chunked_wavs = [
        (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
    ]
    sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
    sequential_chunked_wavs = [
        sequential_indexed_chunked_wavs[i][1]
        for i in range(len(sequential_indexed_chunked_wavs))
    ]
    final_wav = cross_fade_concat(
        sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=True, trail_sil=0
    )
    timings["post_processing"] = (dt.datetime.now() - start).total_seconds()

    timings["total"] = (dt.datetime.now() - total_start).total_seconds()
    timings["wav_seconds"] = final_wav.shape[-1] / sampling_rate

    return timings


def benchmark_configuration(
    model_name: str,
    num_step: int,
    guidance_scale: float,
    prompt_wav_path: str,
    prompt_text: str,
    text: str,
    device: torch.device,
    num_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark a specific configuration."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Configuration: model={model_name}, num_step={num_step}, guidance_scale={guidance_scale}")
    logging.info(f"{'='*60}")

    # Load model
    start = dt.datetime.now()
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, device)
    model_load_time = (dt.datetime.now() - start).total_seconds()
    logging.info(f"Model load time: {model_load_time*1000:.2f} ms")

    # Load vocoder
    start = dt.datetime.now()
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocoder = vocoder.to(device).eval()
    vocoder_load_time = (dt.datetime.now() - start).total_seconds()
    logging.info(f"Vocoder load time: {vocoder_load_time*1000:.2f} ms")

    # Feature extractor
    feature_extractor = VocosFbank()
    sampling_rate = model_config["feature"]["sampling_rate"]

    # Warmup
    logging.info("Running warmup...")
    with torch.no_grad():
        _ = run_single_inference(
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            sampling_rate=sampling_rate,
        )

    # Benchmark runs
    all_timings = []
    for i in range(num_runs):
        logging.info(f"Run {i+1}/{num_runs}...")
        with torch.no_grad():
            timings = run_single_inference(
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                text=text,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                device=device,
                num_step=num_step,
                guidance_scale=guidance_scale,
                sampling_rate=sampling_rate,
            )
            all_timings.append(timings)

    # Calculate averages
    avg_timings = {}
    for key in all_timings[0].keys():
        values = [t[key] for t in all_timings]
        avg_timings[key] = sum(values) / len(values)

    # Add configuration info
    avg_timings["model_name"] = model_name
    avg_timings["num_step"] = num_step
    avg_timings["guidance_scale"] = guidance_scale
    avg_timings["rtf"] = avg_timings["total"] / avg_timings["wav_seconds"]

    return avg_timings


def print_results_table(results: List[Dict]):
    """Print results as a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)

    # Header
    print(f"{'Model':<20} {'Steps':<6} {'CFG':<5} {'Model(ms)':<12} {'Vocoder(ms)':<12} {'Total(ms)':<12} {'RTF':<8} {'Speedup':<8}")
    print("-"*100)

    # Baseline (first result with zipvoice and num_step=16)
    baseline_total = None
    for r in results:
        if r["model_name"] == "zipvoice" and r["num_step"] == 16:
            baseline_total = r["total"]
            break

    if baseline_total is None:
        baseline_total = results[0]["total"]

    for r in results:
        speedup = baseline_total / r["total"] if r["total"] > 0 else 0
        print(
            f"{r['model_name']:<20} "
            f"{r['num_step']:<6} "
            f"{r['guidance_scale']:<5.1f} "
            f"{r['model_inference']*1000:<12.2f} "
            f"{r['vocoder_decode']*1000:<12.2f} "
            f"{r['total']*1000:<12.2f} "
            f"{r['rtf']:<8.4f} "
            f"{speedup:<8.2f}x"
        )

    print("="*100)

    # Detailed breakdown for each configuration
    print("\n" + "="*100)
    print("DETAILED TIMING BREAKDOWN")
    print("="*100)

    for r in results:
        total = r["total"]
        print(f"\n{r['model_name']} (num_step={r['num_step']}, guidance_scale={r['guidance_scale']}):")
        print(f"  Prompt Load:       {r['prompt_load']*1000:8.2f} ms ({r['prompt_load']/total*100:5.1f}%)")
        print(f"  Prompt Preprocess: {r['prompt_preprocess']*1000:8.2f} ms ({r['prompt_preprocess']/total*100:5.1f}%)")
        print(f"  Feature Extract:   {r['feature_extraction']*1000:8.2f} ms ({r['feature_extraction']/total*100:5.1f}%)")
        print(f"  Tokenization:      {r['tokenization']*1000:8.2f} ms ({r['tokenization']/total*100:5.1f}%)")
        print(f"  Model Inference:   {r['model_inference']*1000:8.2f} ms ({r['model_inference']/total*100:5.1f}%)")
        print(f"  Vocoder Decode:    {r['vocoder_decode']*1000:8.2f} ms ({r['vocoder_decode']/total*100:5.1f}%)")
        print(f"  Post Processing:   {r['post_processing']*1000:8.2f} ms ({r['post_processing']/total*100:5.1f}%)")
        print(f"  Total:             {total*1000:8.2f} ms")
        print(f"  Generated Audio:   {r['wav_seconds']:.2f} seconds")
        print(f"  RTF:               {r['rtf']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="ZipVoice Inference Optimization Benchmark"
    )
    parser.add_argument("--prompt-wav", type=str, required=True,
                       help="Path to prompt WAV file")
    parser.add_argument("--prompt-text", type=str, required=True,
                       help="Transcription of prompt WAV")
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of benchmark runs per configuration")
    parser.add_argument("--output-json", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Configurations to test
    configurations = [
        # Baseline: zipvoice with default settings
        {"model_name": "zipvoice", "num_step": 16, "guidance_scale": 1.0},
        # Reduced steps
        {"model_name": "zipvoice", "num_step": 8, "guidance_scale": 1.0},
        {"model_name": "zipvoice", "num_step": 4, "guidance_scale": 1.0},
        # Distilled model with default settings
        {"model_name": "zipvoice_distill", "num_step": 8, "guidance_scale": 3.0},
        # Distilled model with reduced steps
        {"model_name": "zipvoice_distill", "num_step": 4, "guidance_scale": 3.0},
        {"model_name": "zipvoice_distill", "num_step": 2, "guidance_scale": 3.0},
    ]

    results = []
    for config in configurations:
        try:
            result = benchmark_configuration(
                model_name=config["model_name"],
                num_step=config["num_step"],
                guidance_scale=config["guidance_scale"],
                prompt_wav_path=args.prompt_wav,
                prompt_text=args.prompt_text,
                text=args.text,
                device=device,
                num_runs=args.num_runs,
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to benchmark {config}: {e}")
            continue

    # Print results
    print_results_table(results)

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
