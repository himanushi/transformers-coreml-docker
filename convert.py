import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import coremltools as ct

# Paths to the input model and output CoreML model
model_dir = "./models"
output_dir = "./output"
model_name = os.environ["MODEL_NAME"]
print(model_dir, "/", model_name, "/pytorch_model.bin")
model_file = os.path.join(model_dir, model_name, "pytorch_model.bin")
config_file = os.path.join(model_dir, model_name, "config.json")
tokenizer_file = os.path.join(model_dir, model_name, "tokenizer.json")

# Load the model and tokenizer
print("config")
config = AutoConfig.from_pretrained("./models/xlm-roberta-large/config.json")
print("model")
model = AutoModel.from_pretrained("./models/xlm-roberta-large/pytorch_model.bin", config=config)
print("tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./models/xlm-roberta-large")

# Convert the model to PyTorch format
print("cpu")
pytorch_model = model.to('cpu')
pytorch_model.eval()

# Convert the model to CoreML format
print("convert")
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
print("save")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
coreml_model.save(os.path.join(output_dir, f"{model_name}.mlmodel"))

