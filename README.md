# PyTorch Natural Language Processing Project Template
This repository is mainly build upon [victoresque](https://github.com/victoresque/pytorch-template)'s
repository, which is a perfect project for computer vision tasks. However, it was lacking configuration for Natural Language Processing (NLP) tasks, which are obviuosly different from computer vision or time series tasks. Even though there have been many great NLP frameworks published in last years e.g. AllenNLP, fasttext, torchtext, fastai, pytorch-nlp etc., this brought a big complexity and chaos in selection and usage of this frameworks. They all have advantages and disadvantages and one may not easily exploit them togerher easily and has to make a decision. In this project, I aim to provide **a low level api structure** in which any of these frameworks advantage can be exploited easily. For example, in the sample project I used Glove Embeddings by using pytorch-nlp, and what I needed was just to write a simple wrapper class. 

Please enjoy and any feedback is welcomed!!! 

PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch NLP Project Tremplate](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss and metrics](#loss-and-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [TensorboardX Visualization](#tensorboardx-visualization)
	* [Contributing](#contributing)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5
* PyTorch >= 1.0.1.post2
* PyTorch-NLP >= 0.3.7.post1
* tqdm (Optional for `test.py`)
* tensorboard >= 1.13.1 (Optional for TensorboardX)
* tensorboardX >= 1.6 (Optional for TensorboardX)

## Features
* Clear folder structure which is suitable for Natural Language Processing project.
* `.json` config file support for convenient parameter tuning.
* Ability to make use of different NLP frameworks as well as custom methods
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development: 
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.
  * `BaseEmbedding` handles Embedding #TODO

## Folder Structure
  ```bash
  pytorch-nlp-project-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_embedding.py
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │  
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  │
  ├── datasets/ - datasets that will be used in dataloaders
  │
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  │
  ├──embedding/ - embeddings (custom or pretrained any transformer)
  │   └── embedding.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboardX and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboardX visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
  │   ├── util.py
  │   ├── vocab.py
  │   └── ...
  ```

## Usage
The code in this repo is a Language Modelling example with Rick and Morty Episodes data for NLP project template.
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Rick_And_Morty",
  "n_gpu": 1,
  "embedding": {
    "type": "GloveEmbedding",
    "args": {
      "name": "6B",
      "dim": 100
    }
  },
  "arch": {
    "type": "MortyFire",
    "args": {
      "lstm_size": 256,
      "seq_length": 20,
      "num_layers": 2,
      "lstm_dropout": 0.3,
      "fc_dropout": 0.3
    }
  },
  "data_loader": {
    "type": "RickAndMortyDataLoader",
    "args": {
      "data_dir": "data/rick_and_morty",
      "seq_length": 20,
      "vocab_size": 1000,
      "batch_size": 256,
      "shuffle": true,
      "validation_split": 0.2
    }
  },
  "optimizer": {
    "type": "Adam",
    "args": {
      "lr": 0.001,
      "weight_decay": 0,
      "amsgrad": true
    }
  },
  "loss": "cross_entropy",
  "metrics": [
    "accuracy"
  ],
  "lr_scheduler": {
    "type": "StepLR",
    "args": {
      "step_size": 50,
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 20,
    "save_dir": "saved/",
    "save_period": 1,
    "verbosity": 2,
    "monitor": "min val_loss",
    "early_stop": 10,
    "tensorboardX": true
  }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```bash
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```bash
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```bash
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```bash
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.
  
### Embedding
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseEmbedding` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    `TODO`
    Implement the foward pass method `forward()`

* **Example**

  Please refer to `embedding/embedding.py` for a LeNet example.



### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

#### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["my_metric", "my_metric2"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log = {**log, **additional_log}
  return log
  ```
  
### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

```bash
  python test.py --resume path/to/checkpoint.pth
```

### Validation data
In case the validation data is not provided in the dataset, you may use the `split_validation()` from `BaseDataLoader` class. Otherwise, you need to specify the path to the validation dataset and do necessary configuration in `trained.py`. 

To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "Rick_And_Morty",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### TensorboardX Visualization
This template supports [TensorboardX](https://github.com/lanpa/tensorboardX) visualization.
* **TensorboardX Usage**

1. **Install**

    Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Set `tensorboardX` option in config file true.

3. **Open tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`,  etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` module. 

**Note**: You don't have to specify current steps, since `WriterTensorboardX` class defined at `logger/visualization.py` will track current steps.

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs
- [ ] Creating one architecture in which embedding and model are called
- [ ] Iteration-based training (instead of epoch-based)
- [ ] Multiple optimizers
- [ ] Configurable logging layout, checkpoint naming
- [ ] `visdom` logger support

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Pytorch-Template](https://github.com/victoresque/pytorch-template) by [victorresque](https://github.com/victoresque)

**Some Other Repositories worth to mention**

* [fastai](https://github.com/fastai/fastai)
* [AllenNLP](https://github.com/allenai/allennlp)

