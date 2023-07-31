#!/bin/bash -xe

MODEL_NAME=roberta-base

download_file() {
  FILE=$1
  EXPECTED_CHECKSUM=$2

  if [ ! -f "$FILE" ] || [ "$(sha256sum "$FILE" | awk '{ print $1 }')" != "$EXPECTED_CHECKSUM" ]; then
    echo "Downloading $FILE"
    wget -q "https://huggingface.co/$MODEL_NAME/resolve/main/$FILE" -O "$FILE"
    if [ "$(sha256sum "$FILE" | awk '{ print $1 }')" != "$EXPECTED_CHECKSUM" ]; then
      echo "Checksum mismatch for $FILE"
      exit 1
    fi
  fi
}

# Create the directory and navigate into it
mkdir -p $MODEL_NAME
cd $MODEL_NAME

download_file "config.json" "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9"
download_file "merges.txt" "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
download_file "vocab.json" "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655"
download_file "pytorch_model.bin" "278b7a95739c4392fae9b818bb5343dde20be1b89318f37a6d939e1e1b9e461b"
