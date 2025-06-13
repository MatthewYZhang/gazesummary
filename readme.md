# Gaze Summary

## Start

Huggingface is enough? TODO


## Datasets

```
Project Structure
├── data/
│   ├── gaze_stage1/
│   ├── gaze_stage23/
│   ├── raw_split1/           # first portion of synthetic eye gaze data
│   └── raw_split2/           # second portion of synthetic eye gaze data
├── trained_models/           # put trained models here after unzip
├── build_datasets.py         # Script to build datasets for training and testing (synthetic)
├── build_real_datasets.py    # Script to build datasets for testing (real user data)
├── config.py                 # Configuration file for model and training parameters
├── gazeT5.py                 # Main model definition or utilities
├── GazeT5ForCasualLM.py      # GazeT5 adaptation for causal language modeling
├── inference.py              # Script for inference on test data
├── nohup.out                 # Check for desired output for training and evaluation
├── readme.md                 # Project overview and usage instructions
├── stage1_dataset.py         # Convert cleaned datasets from build_datasets.py to training dataset for stage1
├── stage1_train.py           # Training script for Stage 1 (sentence-level)
├── stage2_train.py           # Training script for Stage 2 (document-level), only tune adapter (encoder pooling query, projection layer)
├── stage3_train.py           # Training script for Stage 3 (document-level), lora tune 
└── stage23_dataset.py        # Convert cleaned datasets from build_datasets.py to training dataset for stage2 and stage3
```


## Steps

Download trained models and adapters from [google drive](https://drive.google.com/file/d/17_myYX4n_2_8LPz4_GjUzyyB5I5GRVDw/view?usp=sharing).

First build datasets. This is already finished for this repo, but in case we need new data.

```shell
build_datasets.py
```

Then convert the cleaned datasets to training datasets for stage1 and stage 2 and 3.

```shell
python stage{}_dataset.py
```

Training. Check `train.sh`.
