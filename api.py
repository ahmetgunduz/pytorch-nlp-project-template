import argparse
import os

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

import data_loader.data_loaders as module_data
import embedding.embedding as module_embedding
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import predict_class_from_text

app = Flask(__name__)
CORS(app)
parser = argparse.ArgumentParser(
    description="PyTorch Natural Language Processing Template"
)

parser.add_argument(
    "-r",
    "--resume",
    default=os.path.join(
        "saved", "models", "Email-Spam", "0407_192255", "model_best.pth"
    ),
    type=str,
    help="path to latest checkpoint (default: None)",
)

parser.add_argument(
    "-d",
    "--device",
    default=None,
    type=str,
    help="indices of GPUs to enable (default: all)",
)

config = ConfigParser(parser)
args = parser.parse_args()

# TODO: LOAD VOCAB WITHOUT DATA LOADER

# setup data_loader instances
data_loader = getattr(module_data, config["test_data_loader"]["type"])(
    config["test_data_loader"]["args"]["data_dir"],
    batch_size=32,
    seq_length=128,
    shuffle=False,
    validation_split=0.0,
    training=False,
    num_workers=2,
)

# build model architecture
try:
    config["embedding"]["args"].update({"vocab": data_loader.dataset.vocab})
    embedding = config.initialize("embedding", module_embedding)
except:
    embedding = None
config["arch"]["args"].update({"vocab": data_loader.dataset.vocab})
config["arch"]["args"].update({"embedding": embedding})
model = config.initialize("arch", module_arch)

checkpoint = torch.load(args.resume)
state_dict = checkpoint["state_dict"]
if config["n_gpu"] > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

# prepare model for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    in_text = request.json["in_text"]
    # try:
    score, prediction = predict_class_from_text(
        model=model, input_text=in_text, dataset=data_loader.dataset
    )
    out = prediction.item()
    score = score.item()
    return jsonify({"class": out, "score": score})


if __name__ == "__main__":
    app.run("0.0.0.0", port=8080)
