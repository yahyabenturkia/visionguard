#!/bin/bash
PICK=$(ls ~/visionguard/raw_data/good/*.jpg | shuf -n 1)
NAME=$(basename "$PICK" .jpg)
echo "Testing GOOD: $NAME"
python ~/visionguard/ai/src/inference.py \
  --image "$PICK" \
  --out ~/visionguard/ai/test_outputs/${NAME}_detect.jpg \
  --json ~/visionguard/ai/test_outputs/${NAME}_detect.json
