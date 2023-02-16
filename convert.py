import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import coremltools as ct
from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder



# Paths to the input model and output CoreML model
model_dir = "./models"
output_dir = "./output"
model_name = os.environ["MODEL_NAME"]
print(os.path.join(model_dir, model_name, "pytorch_model.bin"))
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
pytorch_model = model.to('cpu')
pytorch_model.eval()

# Convert the model to CoreML format
print("convert")
# Define the input and output feature of the model
input_features = [('input_ids', ct.SequenceType(ct.Int32TensorType([1, 512]))),
                  ('attention_mask', ct.SequenceType(ct.Int32TensorType([1, 512])))]
output_features = [('classLabel', ct.DictionaryType(ct.int64, ct.float32))]
builder = NeuralNetworkBuilder(input_features, output_features)
builder.add_elementwise(name='output1', input_names=['output'], output_name='output1', mode='RELU')
builder.add_activation(name='output2', non_linearity='LINEAR', input_name='output1', output_name='output2')
builder.add_softmax(name='output3', input_name='output2', output_name='output3')
builder.set_input(input_features[0].name, input_features[0].shape)
builder.set_output(output_features[0].name, output_features[0].shape)
coreml_model = MLModel(builder.spec)

# Save the CoreML model to a file
print("save")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
coreml_model.save(os.path.join(output_dir, f"{model_name}.mlmodel"))
