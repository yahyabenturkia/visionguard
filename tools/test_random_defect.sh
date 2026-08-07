#!/bin/bash
PICK=$(ls ~/visionguard/raw_data/defect/*.jpg | shuf -n 1)
NAME=$(basename "$PICK" .jpg)
echo "Testing DEFECT: $NAME"
python ~/visionguard/ai/src/inference.py \
  --image "$PICK" \
  --out ~/visionguard/ai/test_outputs/${NAME}_detect.jpg \
  --json ~/visionguard/ai/test_outputs/${NAME}_detect.json
