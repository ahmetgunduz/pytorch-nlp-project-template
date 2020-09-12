import json
import logging
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept
        # np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint
            )
        )
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise Exception("File doesn't exist {}")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint


def generate_text(model, start_seq, vocab, length=100, temperature=1.0):
    def _pick_word(probabilities, temperature):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :return: String of the predicted word
        """
        probabilities = np.log(probabilities) / temperature
        exp_probs = np.exp(probabilities)
        probabilities = exp_probs / np.sum(exp_probs)
        pick = np.random.choice(len(probabilities), p=probabilities)
        while int(pick) == 1:
            pick = np.random.choice(len(probabilities), p=probabilities)
        return pick

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = vocab.clean_text(start_seq)
    tokens = vocab.tokenize(tokens)

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, model.seq_length), vocab["<pad>"])
    for idx, token in enumerate(tokens):
        current_seq[-1][idx - len(tokens)] = vocab[token]
    predicted = tokens

    for _ in range(length):
        current_seq = torch.LongTensor(current_seq)
        current_seq = current_seq.to(device)

        output = model((current_seq, None, None))
        p = torch.nn.functional.softmax(output, dim=1).data
        probabilities = p.cpu().numpy().squeeze()

        word_i = _pick_word(probabilities, temperature)

        # retrieve that word from the dictionary
        word = vocab[int(word_i)]
        predicted.append(word)

        # the generated word becomes the next "current sequence" and the cycle
        # can continue
        current_seq = current_seq.cpu().data.numpy()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = " ".join(predicted)
    return gen_sentences

def predict_classes_from_text(model, input_text, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids, attention_mask, segment_ids = dataset.format_in_text(input_text)
    input_ids = input_ids.to(device).unsqueeze(0)
    attention_mask = attention_mask.to(device).unsqueeze(0)
    segment_ids = segment_ids.to(device).unsqueeze(0)
    batch = (input_ids, attention_mask, segment_ids)

    return predict_classes_from_batch(model, batch)

def predict_classes_from_batch(model, batch):
    model.eval()
    output = model(batch=batch)
    return output

def predict_class_from_text(model, input_text, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids, attention_mask, segment_ids = dataset.format_in_text(input_text)
    input_ids = input_ids.to(device).unsqueeze(0)
    attention_mask = attention_mask.to(device).unsqueeze(0)
    segment_ids = segment_ids.to(device).unsqueeze(0)
    batch = (input_ids, attention_mask, segment_ids)

    return predict_class_from_batch(model, batch)

def predict_class_from_batch(model, batch):
    m = torch.nn.Softmax(dim=1)
    model.eval()
    output = model(batch=batch)
    output = m(output)
    return torch.max(output, 1)


class SimpleTokenizer(object):
    def __init__(self):
        print("Simple tokenizer defined")
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3

    def tokenize(self, text):
        return text.split()
