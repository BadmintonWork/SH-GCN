# SH-GCN
This project is generated based on the the pyskl (https://github.com/kennymckormick/pyskl)

## Installation
```shell

git clone https://github.com/BadmintonWork/SH-GCN
cd SH-GCN

conda env create -f pyskl.yaml

conda activate pyskl

pip install -e .

```

## Data Preparation

This repository utilizes ShuttleSet and VideoBadminton as raw video sources. We have developed two refined multimodal versions: ShuttleSet-MM and VideoBadminton-MM. These datasets are derived and augmented from the originals to incorporate both skeletal topology and shuttlecock trajectories.

Note: The full datasets will be publicly released on this platform once the paper is officially accepted.

## Training & Testing

You can use following commands for training and testing.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
