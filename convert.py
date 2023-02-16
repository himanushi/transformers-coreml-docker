import os
import torch
from transformers import AutoModel, AutoTokenizer
import coremltools as ct

# Paths to the input model and output CoreML model
model_dir = "models"
output_dir = "output"
model_file = os.path.join(model_dir, "pytorch_model.bin")
config_file = os.path.join(model_dir, "config.json")
tokenizer_file = os.path.join(model_dir, "tokenizer.json")

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)

# Convert the model to PyTorch format
pytorch_model = model.to('cpu')
pytorch_model.eval()

# Convert the model to CoreML format
coreml_model = ct.convert(
    pytorch_model,
    inputs=[ct.ImageType()],
    mode="classifier",
    predicted_feature_name="class",
    minimum_ios_deployment_target='13',
    image_input_names='input',
    image_scale=1/255.0,
    red_bias=0,
    green_bias=0,
    blue_bias=0
)

# Save the CoreML model to a file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
coreml_model.save(os.path.join(output_dir, "model.mlmodel"))
