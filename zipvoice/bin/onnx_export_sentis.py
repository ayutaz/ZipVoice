#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
#                   2025  Unity Sentis Export  (modified for Unity Sentis compatibility)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script exports a pre-trained ZipVoice or ZipVoice-Distill model from PyTorch to
ONNX format optimized for Unity Sentis.

Key differences from standard onnx_export.py:
- Uses opset version 15 (recommended for Sentis)
- Enables constant folding for optimization
- Adds Sentis compatibility metadata
- Includes verification step for Sentis-unsupported operators

Usage:

python3 -m zipvoice.bin.onnx_export_sentis \
    --model-name zipvoice \
    --model-dir exp/zipvoice \
    --checkpoint-name epoch-11-avg-4.pt \
    --onnx-model-dir exp/zipvoice_sentis

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.
"""


import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import onnx
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled

# HuggingFace repository for pre-trained models
HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}

# Operators known to be unsupported by Unity Sentis
SENTIS_UNSUPPORTED_OPS: Set[str] = {
    "If",  # Conditional control flow - use static graphs instead
    "Log1p",  # Use Log(1+x) instead
    "LogAddExp",
    "ComplexAbs",
    "FFT",
    "IFFT",
    "RFFT",
    "IRFFT",
    "Unique",
    "StringNormalizer",
    "TfIdfVectorizer",
    "Tokenizer",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--onnx-model-dir",
        type=str,
        default="exp/zipvoice_sentis",
        help="Dir to the exported models (default: exp/zipvoice_sentis)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model.pt",
        help="The name of model checkpoint.",
    )

    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip INT8 quantization (faster export)",
    )

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


def _precompute_positional_encodings(model: nn.Module, max_len: int) -> None:
    """Pre-compute positional encodings for all encoder_pos modules.

    This function traverses the model and extends the positional encodings
    to the specified maximum length. This is necessary for ONNX export to
    avoid the conditional "If" operator that Unity Sentis doesn't support.

    Args:
        model: The ZipVoice model.
        max_len: Maximum sequence length to pre-compute.
    """
    dummy_input = torch.zeros(max_len)

    # Find and extend all CompactRelPositionalEncoding modules
    for name, module in model.named_modules():
        if hasattr(module, 'extend_pe') and hasattr(module, 'pe'):
            module.extend_pe(dummy_input)
            if module.pe is not None:
                logging.info(f"  {name}: PE shape = {module.pe.shape}")


def verify_sentis_compatibility(model_path: str) -> List[str]:
    """Verify that the ONNX model is compatible with Unity Sentis.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        List of warning messages for unsupported operators.
    """
    model = onnx.load(model_path)
    warnings = []

    # Check for unsupported operators
    for node in model.graph.node:
        if node.op_type in SENTIS_UNSUPPORTED_OPS:
            warnings.append(
                f"WARNING: Operator '{node.op_type}' (node: {node.name}) "
                f"is not supported by Unity Sentis"
            )

    # Check tensor dimensions (Sentis limit: 8 dimensions)
    for value_info in model.graph.value_info:
        if value_info.type.HasField("tensor_type"):
            dims = len(value_info.type.tensor_type.shape.dim)
            if dims > 8:
                warnings.append(
                    f"ERROR: Tensor '{value_info.name}' has {dims} dimensions "
                    f"(Sentis limit: 8)"
                )

    # Check input/output tensor dimensions
    for io_info in list(model.graph.input) + list(model.graph.output):
        if io_info.type.HasField("tensor_type"):
            dims = len(io_info.type.tensor_type.shape.dim)
            if dims > 8:
                warnings.append(
                    f"ERROR: I/O tensor '{io_info.name}' has {dims} dimensions "
                    f"(Sentis limit: 8)"
                )

    return warnings


class OnnxTextModel(nn.Module):
    def __init__(self, model: nn.Module):
        """A wrapper for ZipVoice text encoder."""
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id

    def forward(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tensor:
        cat_tokens = torch.cat([prompt_tokens, tokens], dim=1)
        cat_tokens = nn.functional.pad(cat_tokens, (0, 1), value=self.pad_id)
        tokens_len = cat_tokens.shape[1] - 1
        padding_mask = (torch.arange(tokens_len + 1) == tokens_len).unsqueeze(0)

        embed = self.embed(cat_tokens)
        embed = self.text_encoder(x=embed, t=None, padding_mask=padding_mask)

        features_len = torch.ceil(
            (prompt_features_len / prompt_tokens.shape[1] * tokens_len / speed)
        ).to(dtype=torch.int64)

        token_dur = torch.div(features_len, tokens_len, rounding_mode="floor").to(
            dtype=torch.int64
        )

        # Convert rank-1 tensor to rank-0 tensor for ONNX compatibility
        token_dur = token_dur.reshape(())
        features_len = features_len.reshape(())

        text_condition = embed[:, :-1, :].unsqueeze(2).expand(-1, -1, token_dur, -1)
        text_condition = text_condition.reshape(embed.shape[0], -1, embed.shape[2])

        text_condition = torch.cat(
            [
                text_condition,
                embed[:, -1:, :].expand(-1, features_len - text_condition.shape[1], -1),
            ],
            dim=1,
        )

        return text_condition


class OnnxFlowMatchingModel(nn.Module):
    def __init__(self, model: nn.Module, distill: bool = False):
        """A wrapper for ZipVoice flow-matching decoder."""
        super().__init__()
        self.distill = distill
        self.fm_decoder = model.fm_decoder
        self.model_func = getattr(model, "forward_fm_decoder")
        self.feat_dim = model.feat_dim

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                guidance_scale=guidance_scale,
            )
        else:
            x = x.repeat(2, 1, 1)
            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )
            speech_condition = torch.cat(
                [
                    torch.where(
                        t > 0.5, torch.zeros_like(speech_condition), speech_condition
                    ),
                    speech_condition,
                ],
                dim=0,
            )
            guidance_scale = torch.where(t > 0.5, guidance_scale, guidance_scale * 2.0)
            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
            ).chunk(2, dim=0)
            v = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return v


def export_text_encoder(
    model: OnnxTextModel,
    filename: str,
    opset_version: int = 15,
) -> None:
    """Export the text encoder model to ONNX format for Unity Sentis.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use (default: 15 for Sentis).
    """
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
    prompt_tokens = torch.tensor([[0, 1]], dtype=torch.int64)
    prompt_features_len = torch.tensor(10, dtype=torch.int64)
    speed = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(model, (tokens, prompt_tokens, prompt_features_len, speed))

    torch.onnx.export(
        model,
        (tokens, prompt_tokens, prompt_features_len, speed),
        filename,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,  # Enable constant folding for optimization
        input_names=["tokens", "prompt_tokens", "prompt_features_len", "speed"],
        output_names=["text_condition"],
        dynamic_axes={
            "tokens": {0: "N", 1: "T"},
            "prompt_tokens": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice text encoder (Unity Sentis compatible)",
        "use_espeak": "1",
        "use_pinyin": "1",
        "target_runtime": "unity_sentis",
        "opset_version": str(opset_version),
    }
    logging.info(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)

    # Verify Sentis compatibility
    warnings = verify_sentis_compatibility(filename)
    if warnings:
        for w in warnings:
            logging.warning(w)
    else:
        logging.info("Text encoder passed Sentis compatibility check")

    logging.info(f"Exported to {filename}")


def export_fm_decoder(
    model: OnnxFlowMatchingModel,
    filename: str,
    opset_version: int = 15,
) -> None:
    """Export the flow matching decoder model to ONNX format for Unity Sentis.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use (default: 15 for Sentis).
    """
    feat_dim = model.feat_dim
    # Use longer sequence length for tracing to ensure dynamic shapes work correctly
    # The PE is precomputed to max_pe_len=4000, so trace with a length that covers
    # typical use cases (up to ~2500 frames for ~100 seconds of audio)
    seq_len = 2500
    t = torch.tensor(0.5, dtype=torch.float32)
    x = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    text_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    speech_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    guidance_scale = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(
        model, (t, x, text_condition, speech_condition, guidance_scale)
    )

    torch.onnx.export(
        model,
        (t, x, text_condition, speech_condition, guidance_scale),
        filename,
        verbose=False,
        opset_version=opset_version,
        do_constant_folding=True,  # Enable constant folding for optimization
        input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale"],
        output_names=["v"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
            "speech_condition": {0: "N", 1: "T"},
            "v": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice flow-matching decoder (Unity Sentis compatible)",
        "feat_dim": str(feat_dim),
        "sample_rate": "24000",
        "n_fft": "1024",
        "hop_length": "256",
        "window_length": "1024",
        "num_mels": "100",
        "target_runtime": "unity_sentis",
        "opset_version": str(opset_version),
    }
    logging.info(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)

    # Verify Sentis compatibility
    warnings = verify_sentis_compatibility(filename)
    if warnings:
        for w in warnings:
            logging.warning(w)
    else:
        logging.info("FM decoder passed Sentis compatibility check")

    logging.info(f"Exported to {filename}")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))

    if params.model_dir is not None:
        # Use local model directory
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")
        for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
        logging.info(f"Loading model from local directory: {params.model_dir}")
    else:
        # Download from HuggingFace
        logging.info(f"Downloading {params.model_name} model from HuggingFace ({HUGGINGFACE_REPO})")
        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt"
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )
        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )
        logging.info(f"Downloaded model files from HuggingFace")

    tokenizer = SimpleTokenizer(token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = False
    else:
        assert params.model_name == "zipvoice_distill"
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = True

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    # Pre-compute positional encodings to a sufficient length for ONNX export
    # This avoids the conditional "If" operator in the exported graph
    max_pe_len = 4000  # ~170 seconds at 24kHz with hop_length=256
    logging.info(f"Pre-computing positional encodings to max_len={max_pe_len}")
    _precompute_positional_encodings(model, max_pe_len)

    logging.info("Exporting model for Unity Sentis")
    onnx_model_dir = Path(params.onnx_model_dir)
    onnx_model_dir.mkdir(parents=True, exist_ok=True)

    # Use opset version 15 for Unity Sentis (supported range: 7-15)
    opset_version = 15

    text_encoder = OnnxTextModel(model=model)
    text_encoder_file = onnx_model_dir / "text_encoder.onnx"
    export_text_encoder(
        model=text_encoder,
        filename=text_encoder_file,
        opset_version=opset_version,
    )

    fm_decoder = OnnxFlowMatchingModel(model=model, distill=distill)
    fm_decoder_file = onnx_model_dir / "fm_decoder.onnx"
    export_fm_decoder(
        model=fm_decoder,
        filename=fm_decoder_file,
        opset_version=opset_version,
    )

    if not params.skip_quantization:
        logging.info("Generate int8 quantization models")

        text_encoder_int8_file = onnx_model_dir / "text_encoder_int8.onnx"
        quantize_dynamic(
            model_input=text_encoder_file,
            model_output=text_encoder_int8_file,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

        fm_decoder_int8_file = onnx_model_dir / "fm_decoder_int8.onnx"
        quantize_dynamic(
            model_input=fm_decoder_file,
            model_output=fm_decoder_int8_file,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
    else:
        logging.info("Skipping INT8 quantization")

    logging.info("=" * 60)
    logging.info("Export completed successfully!")
    logging.info(f"Output directory: {onnx_model_dir}")
    logging.info("Files exported:")
    logging.info(f"  - {text_encoder_file}")
    logging.info(f"  - {fm_decoder_file}")
    if not params.skip_quantization:
        logging.info(f"  - {onnx_model_dir / 'text_encoder_int8.onnx'}")
        logging.info(f"  - {onnx_model_dir / 'fm_decoder_int8.onnx'}")
    logging.info("=" * 60)
    logging.info("Next steps for Unity:")
    logging.info("  1. Copy ONNX files to Unity Assets/Models/")
    logging.info("  2. Download Vocos ONNX: python scripts/download_vocos_onnx.py")
    logging.info("  3. Implement EulerSolver and ISTFT in C#")
    logging.info("=" * 60)


if __name__ == "__main__":

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
