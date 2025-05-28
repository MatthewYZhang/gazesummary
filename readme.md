# Gaze Summary

## Start



## Models


## Datasets

```
Project Structure
├── data/
│   ├── gaze_stage1/
│   ├── gaze_stage23/
│   ├── raw_split1/           # first portion of synthetic eye gaze data
│   └── raw_split2/           # second portion of synthetic eye gaze data
├── build_datasets.py         # Script to build datasets for training
├── config.py                 # Configuration file for model and training parameters
├── gazeT5.py                 # Main model definition or utilities
├── GazeT5ForCasualLM.py      # GazeT5 adaptation for causal language modeling
├── readme.md                 # Project overview and usage instructions
├── train_stage_1.py          # Training script for Stage 1 (sentence-level)
└── train_stage_2.py          # Training script for Stage 2/3 (document-level)
```


## Training

