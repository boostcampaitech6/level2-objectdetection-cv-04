# AI Tech 6기 Team 아웃라이어

## Members
- 강승환
- 김승민
- 설훈
- 전병관
- 조성혁
- 이도형


## 재활용 품목 분류를 위한 Object Detection

요즘 쓰레기 관련 문제들이 항상 문제로 떠오르고 있습니다. 이러한 문제를 해결할 수 있는 방법 중 하나는 분리수거를 잘 하는 것입니다. 잘 분리 배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리 배출 되면 그대로 폐기물로 분류되어 매립 또는 소각 되기 때문입니다. 따라서 분리 배출을 잘 하기 위해서 이미지에서 쓰레기를 detection하는 모델을 만들어 보려고 합니다. 쓰레기를 잘 detection하는 모델을 만든다면 쓰레기장에 설치되어 정확한 분리수거를 도울 수 있고 어린 아이들의 분리수거 교육에 사용될 수도 있을 것입니다.

이번 프로젝트는 부스트캠프 AI Tech CV 트랙내에서 진행된 대회이며 mAP50으로 최종평가를 진행하게 됩니다.

## Ground Rules
### [Conventional Commits 1.0.0](https://www.conventionalcommits.org/ko/v1.0.0/)
```bash
<타입>[적용 범위(선택 사항)]: <설명>

[본문(선택 사항)]

[꼬리말(선택 사항)]
```

#### Types
- fix | feat | BREAKING CHANGE | build | chore | ci | docs | style | refactor | test | release
  - fix : 기능에 대한 버그 수정
  - feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
  - build : 빌드 관련 수정
  - chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
  - ci : CI 관련 설정 수정
  - docs : 문서(주석) 수정
  - style : 코드 스타일, 포맷팅에 대한 수정
  - refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
  - test : 테스트 코드 추가/수정
  - release : 버전 릴리즈

### Github Flow

> It will be updated.

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format.
Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
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

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

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

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
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

### Wandb Visualization
This template supports Wandb visualization by using [Wandb](https://github.com/wandb/wandb) library.

#### Quickstart

Get started with W&B in four steps:

1. First, sign up for a [free W&B account](https://wandb.ai/login?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=quickstart).

2. Second, install the W&B SDK with [pip](https://pip.pypa.io/en/stable/). Navigate to your terminal and type the following command:

```bash
pip install wandb
```

3. Third, log into W&B:

```python
wandb.login()
```

4. Use the example code snippet below as a template to integrate W&B to your Python script:

```python
import wandb

# Start a W&B Run with wandb.init
run = wandb.init(project="my_first_project")

# Save model inputs and hyperparameters in a wandb.config object
config = run.config
config.learning_rate = 0.01

# Model training code here ...

# Log metrics over time to visualize performance with wandb.log
for i in range(10):
    run.log({"loss": loss})
```

That's it! Navigate to the W&B App to view a dashboard of your first W&B Experiment. Use the W&B App to compare multiple experiments in a unified place, dive into the results of a single run, and much more!

<p align='center'>
<img src="https://github.com/wandb/wandb/blob/main/docs/README_images/wandb_demo_experiments.gif?raw=true" width="100%">
</p>
<p align = "center">
Example W&B Dashboard that shows Runs from an Experiment.
</p>

&nbsp;

## Contribution

- 

## TODOs

- [ ] 

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
