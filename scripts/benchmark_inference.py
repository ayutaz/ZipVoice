#!/usr/bin/env python3
"""
Detailed inference timing benchmark script.
Measures time spent in each component of the ZipVoice inference pipeline.
"""

import datetime as dt
import json
import sys
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
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
from zipvoice.vocoder import get_vocoder


def benchmark_inference(
    prompt_wav_path: str,
    prompt_text: str,
    text: str,
    vocoder_type: str = "vocos",
    vocoder_n_steps: int = 2,
    num_runs: int = 3,
):
    """Run detailed timing benchmark."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampling_rate = 24000
    num_step = 16
    guidance_scale = 1.0
    speed = 1.0
    t_shift = 0.5
    target_rms = 0.1
    feat_scale = 0.1

    timings = {
        "model_load": [],
        "vocoder_load": [],
        "prompt_load": [],
        "prompt_preprocess": [],
        "feature_extraction": [],
        "tokenization": [],
        "model_inference": [],
        "vocoder_decode": [],
        "post_processing": [],
        "total": [],
    }

    print(f"Device: {device}")
    print(f"Vocoder: {vocoder_type}" + (f" ({vocoder_n_steps}-step)" if vocoder_type == "flow2gan" else ""))
    print(f"Running {num_runs} iterations...")
    print()

    # === Model Loading (only once) ===
    start = dt.datetime.now()

    # Download model
    model_dir = "zipvoice"
    checkpoint_path = hf_hub_download(
        repo_id="k2-fsa/ZipVoice",
        filename=f"{model_dir}/model.pt",
    )
    config_path = hf_hub_download(
        repo_id="k2-fsa/ZipVoice",
        filename=f"{model_dir}/model.json",
    )
    tokens_path = hf_hub_download(
        repo_id="k2-fsa/ZipVoice",
        filename=f"{model_dir}/tokens.txt",
    )

    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Initialize tokenizer first to get vocab_size and pad_id
    tokenizer = EspeakTokenizer(token_file=tokens_path, lang="en-us")
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )
    load_checkpoint(checkpoint_path, model)
    model = model.to(device).eval()

    timings["model_load"].append((dt.datetime.now() - start).total_seconds())

    # === Vocoder Loading ===
    start = dt.datetime.now()

    if vocoder_type == "flow2gan":
        vocoder = get_vocoder("flow2gan", n_timesteps=vocoder_n_steps)
    else:
        vocoder = get_vocoder("vocos")
    vocoder = vocoder.to(device).eval()

    timings["vocoder_load"].append((dt.datetime.now() - start).total_seconds())

    # === Feature Extractor ===
    feature_extractor = VocosFbank()

    # Warmup run
    print("Warmup run...")
    with torch.no_grad():
        _ = run_inference(
            prompt_wav_path, prompt_text, text,
            model, vocoder, tokenizer, feature_extractor,
            device, num_step, guidance_scale, speed, t_shift,
            target_rms, feat_scale, sampling_rate,
            timings=None  # Don't record warmup
        )

    # Benchmark runs
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        with torch.no_grad():
            run_timings = run_inference(
                prompt_wav_path, prompt_text, text,
                model, vocoder, tokenizer, feature_extractor,
                device, num_step, guidance_scale, speed, t_shift,
                target_rms, feat_scale, sampling_rate,
                timings={}
            )

            for key, value in run_timings.items():
                timings[key].append(value)

    # Calculate averages
    print("\n" + "=" * 60)
    print("DETAILED INFERENCE TIMING RESULTS")
    print("=" * 60)

    results = {}
    total_time = sum(timings["total"]) / len(timings["total"])

    for key, values in timings.items():
        if key in ["model_load", "vocoder_load"]:
            continue  # Skip one-time costs
        if values:
            avg = sum(values) / len(values)
            percentage = (avg / total_time) * 100 if total_time > 0 else 0
            results[key] = {"avg_seconds": avg, "percentage": percentage}
            print(f"{key:25s}: {avg*1000:8.2f} ms ({percentage:5.1f}%)")

    print("-" * 60)
    print(f"{'Total':25s}: {total_time*1000:8.2f} ms (100.0%)")
    print()

    # One-time costs
    print("One-time costs (not included in total):")
    print(f"  Model load:   {timings['model_load'][0]*1000:8.2f} ms")
    print(f"  Vocoder load: {timings['vocoder_load'][0]*1000:8.2f} ms")

    return results


def run_inference(
    prompt_wav_path, prompt_text, text,
    model, vocoder, tokenizer, feature_extractor,
    device, num_step, guidance_scale, speed, t_shift,
    target_rms, feat_scale, sampling_rate,
    timings=None,
):
    """Run single inference with detailed timing."""

    if timings is None:
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

    torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    return timings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-wav", type=str, default="data/prompt.wav")
    parser.add_argument("--prompt-text", type=str, default="Hello, this is a test.")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--vocoder-type", type=str, default="vocos", choices=["vocos", "flow2gan"])
    parser.add_argument("--vocoder-n-steps", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()

    benchmark_inference(
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        text=args.text,
        vocoder_type=args.vocoder_type,
        vocoder_n_steps=args.vocoder_n_steps,
        num_runs=args.num_runs,
    )
