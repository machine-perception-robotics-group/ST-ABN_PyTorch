# Spatio-Temporal Attention Branch Network for Action Recognition implemented by PyTorch

Writer : Masahiro Mitsuhara

<!--Maintainer: [Tsubasa Hirakawa](https://thirakawa.github.io)-->

This repository is PyTorch implementation of Spatio-Temporal Attention Branch Network.
Our source code is based on [TPN](https://github.com/decisionforce/TPN) and [MMAction](https://github.com/open-mmlab/mmaction) implemented with PyTorch. We are grateful for the author!

## Enviroment
Our source code corresponds to the latest version of PyTorch.
Requirements of PyTorch version are as follows:

- PyTorch: 1.7.0
- torchvision: 0.7.0
- Python: 3.5+
- NVCC: 2+
- GCC: 4.9+
- mmcv: 0.2.10

### Install MMAction
#### Install Cython
```shell
pip install cython
```
#### Install mmaction
```shell
python setup.py develop
```

### Docker
We are using the PyTorch docker image published by NVIDIA.
For more information, please see [here](https://www.nvidia.com/ja-jp/gpu-cloud/containers/).
<!--我々は，NVIDIAが公開しているPyTorchのdockerイメージを使用しています．
詳細は，[こちら](https://www.nvidia.com/ja-jp/gpu-cloud/containers/)をご覧ください．-->

## Data Preparation

### Notes on Video Data format
Since the original VideoDataloader of MMAction requires [decord](https://github.com/zhreshold/decord) for efficient video loading which is non-trivial to compile, this repo only supports **raw frame** format of videos. Therefore, you have to extract frames from raw videos. We will find another libaries and support VideoLoader soon.

### Supported datasets
The `rawframe_dataset` loads data in a general manner by preparing a `.txt` file which contains the directory path of frames, total number of a certain video, and the groundtruth label. After that, specify the `data_root` and `image_tmpl` of config files.


### Prepare annotations

- [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) contains ~240k training videos and ~19k validation videos. See the [guide](https://github.com/open-mmlab/mmaction/tree/master/data_tools/kinetics400/PREPARING_KINETICS400.md) of original MMAction to generate annotations.
- [Something-Someting](https://github.com/TwentyBN) has 2 versions which you have to apply on their [website](https://20bn.com/datasets/something-something). See the [guide](https://github.com/mit-han-lab/temporal-shift-module/tree/master/tools) of TSM to generate annotations.

Thank original [MMAction](https://github.com/open-mmlab/mmaction) and [TSM](https://github.com/mit-han-lab/temporal-shift-module) repo for kindly providing preprocessing scripts.


## Training
 
Our codebase also supports distributed training and non-distributed training.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
```python
evaluation = dict(interval=10)  # This evaluate the model per 10 epoch.
```

### Train with a single GPU
```shell
python tools/train_recognizer.py ${CONFIG_FILE}

#Example of run command is as follows:
python tools/train_recognizer.py config_files/sthv1/st_abn_32.py --validate --work_dir checkpoints/results --gpus 1
```
If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs
```shell
./tools/dist_train_recognizer.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]

#Example of run command is as follows:
./tools/dist_train_recognizer.sh config_files/sthv1/st_abn_32.py 8 --validate --work_dir checkpoints/results
```

Optional arguments:
- `--validate`: Perform evaluation at every 1 epoch during the training.
- `--work_dir`: All outputs (log files and checkpoints) will be saved to the working directory. 
- `--resume_from`: Resume from a previous checkpoint file.
- `--load_from`: Only loads the model weights and the training epoch starts from 0.
 
Difference between `resume_from` and `load_from`: `resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. `load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

**Important**: The default learning rate in config files is for 8 GPUs and 8 video/gpu (batch size = 8*8 = 64). According to the Linear Scaling Rule, you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 8 GPUs * 8 video/gpu and lr=0.04 for 32 GPUs * 8 video/gpu.

## Evaluation
Our codebase supports distributed and non-distributed evaluation mode for reference model. Actually, distributed testing is a little faster than non-distributed testing.  
```
# non-distributed testing
python tools/test_recognizer.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] {--gpus ${GPU_NUM}} --ignore_cache

# distributed testing
./tools/dist_test_recognizer.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] --ignore_cache
```
Optional arguments:
- `--ignore_cache`: If specified, the results cache will be ignored.
- `-s`: If you want to visualize the attention map, add it as an option.

**Important**: The results may vary between distributed evaluation mode and non-distributed evaluation mode. It is recommended to use distributed evaluation mode for evaluation.
To make the attention map visible, uncomment the last line of `def forward_test()` in TSN3D.py.
When learning, comment out everything except `rx`, which is the output of the perception branch.

<!--distributed evaluation modeとnon-distributed evaluation modeで結果が変化する場合があります．distributed evaluation modeを使用して評価することを推奨します．
Attention mapを可視化する際は，TSN3D.pyの`def forward_test()`において，最終行のコメントアウトを解除してください．
学習時はperception branchの出力である`rx`以外はコメントアウトしてください．-->

Examples:
Assume that you have already saved the checkpoints to the directory `checkpoints/`.

1. Test model with non-distributed evaluation mode on 8 GPUs
```
python tools/test_recognizer.py config_files/sthv1/st_abn_32.py checkpoints/results/epoch_$$.pth --gpus 8 --out result.pkl --ignore_cache
```
2. Test model with distributed evaluation mode on 8 GPUs
```shell
./tools/dist_test_recognizer.sh config_files/sthv1/st_abn_32.py checkpoints/results/epoch_$$.pth 8 --out result.pkl --ignore_cache
```
