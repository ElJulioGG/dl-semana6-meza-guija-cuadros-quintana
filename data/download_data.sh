#!/usr/bin/env bash
# Downloads the Blood Cell Images dataset from Kaggle.
#
# Requirements:
#   - Kaggle API token at ~/.kaggle/kaggle.json
#     Get it from: https://www.kaggle.com/settings → API → Create New Token
#   - pip install kaggle
#
# Usage:
#   bash data/download_data.sh

set -euo pipefail

DATASET="paultimothymooney/blood-cells"
OUT_DIR="$(dirname "$0")"   # saves into data/

echo "Downloading dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$OUT_DIR" --unzip

echo "Done. Dataset saved to: $OUT_DIR"
echo "Expected structure:"
echo "  data/dataset2-master/dataset2-master/images/TRAIN/<class>/"
echo "  data/dataset2-master/dataset2-master/images/TEST/<class>/"
