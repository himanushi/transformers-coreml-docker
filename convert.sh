#!/bin/bash

# Set the model name
MODEL_NAME="$1"

# Check if the model directory exists
if [ ! -d "./models/$MODEL_NAME" ]; then
    echo "Error: Model directory ./models/$MODEL_NAME does not exist"
    exit 1
fi

# Check if the pytorch_model.bin file exists
if [ ! -f "./models/$MODEL_NAME/pytorch_model.bin" ]; then
    echo "Error: File ./models/$MODEL_NAME/pytorch_model.bin does not exist"
    exit 1
fi

# Set the input and output directories
INPUT_DIR="$(pwd)/models/$MODEL_NAME"
OUTPUT_DIR="$(pwd)/output"

# Run the Docker container
docker run --rm -v "$INPUT_DIR":/app/models -v "$OUTPUT_DIR":/app/output transformer2coreml