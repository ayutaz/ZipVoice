#!/usr/bin/env python3
"""Convert Windows paths to Linux paths in manifest files."""
import gzip
import sys

def convert_manifest(manifest_path):
    print(f'Converting paths in {manifest_path}...')
    lines = []
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as f:
        for line in f:
            # Replace Windows backslashes with forward slashes
            # Handle both escaped (\\) and single (\) backslashes
            line = line.replace('\\\\', '/').replace('\\', '/')
            lines.append(line)

    with gzip.open(manifest_path, 'wt', encoding='utf-8') as f:
        f.writelines(lines)

    print(f'  Done!')

if __name__ == '__main__':
    manifests = [
        'data/fbank/tsukuyomi_cuts_train.jsonl.gz',
        'data/fbank/tsukuyomi_cuts_dev.jsonl.gz',
    ]
    for manifest in manifests:
        convert_manifest(manifest)
    print('All manifests converted')
