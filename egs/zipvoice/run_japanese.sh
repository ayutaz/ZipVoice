#!/bin/bash

# This script is for fine-tuning ZipVoice on Japanese datasets.
# Example: Tsukuyomi-chan Corpus Vol.1 (30 minutes, 100 sentences)

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=8

# Number of jobs for data preparation
nj=4

# Tokenizer settings for Japanese
tokenizer=japanese

# Maximum length (seconds) of the training utterance
max_len=20

# Download directory for pre-trained models
download_dir=download/

# Training parameters
num_iters=10000
base_lr=0.0001
max_duration=500
save_every_n=1000

# wandb settings (enabled by default)
use_wandb=true
wandb_project=zipvoice-japanese

# Tsukuyomi corpus path (modify this to your corpus location)
# Expected format:
#   - Transcripts: {corpus_dir}/04 台本と補足資料/★台本テキスト/01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt
#   - Audio: {corpus_dir}/02 WAV（+12dB増幅）/*.wav
tsukuyomi_corpus_dir=""

### Prepare TSV from Tsukuyomi corpus (Stage 0)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
      echo "Stage 0: Prepare TSV from Tsukuyomi-chan corpus"

      [ -z "$tsukuyomi_corpus_dir" ] && { echo "Error: tsukuyomi_corpus_dir is not set!" >&2; exit 1; }

      uv run python ./local/prepare_tsv_tsukuyomi.py \
            --corpus-dir "${tsukuyomi_corpus_dir}" \
            --output data/raw/tsukuyomi_train.tsv

      # Copy train to dev for validation (small dataset)
      cp data/raw/tsukuyomi_train.tsv data/raw/tsukuyomi_dev.tsv
fi

### Prepare the training data (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for Japanese dataset from tsv files"

      for subset in train dev; do
            file_path=data/raw/tsukuyomi_${subset}.tsv
            [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
      done

      for subset in train dev; do
            uv run python -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/tsukuyomi_${subset}.tsv \
                  --prefix tsukuyomi \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # The output manifest files are "data/manifests/tsukuyomi_cuts_raw_train.jsonl.gz"
      # and "data/manifests/tsukuyomi_cuts_raw_dev.jsonl.gz".
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests using Japanese tokenizer"

      for subset in train dev; do
            uv run python -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/tsukuyomi_cuts_raw_${subset}.jsonl.gz \
                  --output-file data/manifests/tsukuyomi_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer}
      done
      # The output manifest files are "data/manifests/tsukuyomi_cuts_train.jsonl.gz"
      # and "data/manifests/tsukuyomi_cuts_dev.jsonl.gz".
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Compute Fbank for Japanese dataset"

      for subset in train dev; do
            uv run python -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset tsukuyomi \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download pre-trained model and config"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Prepare Japanese tokens file (Stage 5)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Prepare Japanese tokens file"

      uv run python ./local/prepare_tokens_japanese.py \
            --output data/tokens_japanese.txt
fi

### Training ZipVoice (6 - 7)

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Fine-tune the ZipVoice model for Japanese"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      wandb_args=""
      if [ "$use_wandb" = true ]; then
            wandb_args="--wandb 1 --wandb-project ${wandb_project}"
      else
            wandb_args="--no-wandb"
      fi

      uv run python -m zipvoice.bin.train_zipvoice \
            --world-size 1 \
            --use-fp16 1 \
            --finetune 1 \
            --base-lr ${base_lr} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every_n} \
            --max-duration ${max_duration} \
            --max-len ${max_len} \
            --model-config ${download_dir}/zipvoice/model.json \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --tokenizer ${tokenizer} \
            --token-file data/tokens_japanese.txt \
            --dataset tsukuyomi \
            --train-manifest data/fbank/tsukuyomi_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/tsukuyomi_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_japanese \
            ${wandb_args}
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Average the checkpoints for ZipVoice"
      uv run python -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_japanese
      # The generated model is exp/zipvoice_japanese/iter-10000-avg-2.pt
fi

### Inference with PyTorch models (8)

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "Stage 8: Inference of the Japanese ZipVoice model"

      uv run python -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_japanese/ \
            --checkpoint-name iter-${num_iters}-avg-2.pt \
            --tokenizer ${tokenizer} \
            --token-file data/tokens_japanese.txt \
            --test-list test.tsv \
            --res-dir results/test_japanese \
            --num-step 16
fi
