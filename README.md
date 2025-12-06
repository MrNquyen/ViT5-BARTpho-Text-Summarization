# Implementation for Text Summarization using **ViT5** and **BARTpho**

## **Data Preparation**
- Original Text: Extract PaddleOCR and VietOCR


## Update Modules
```
  Completed
```

## Setup
Setup model using
```
  ./scripts/setup.sh
```

Or:
```
  python setup.py build_ext --inplace
```

## Training
Training on your own dataset:
```
  ./scripts/train.sh
```

Or:
```
  python main.py \
    --config ./config/<vit5/bart>_config.yaml \
    --save_dir ./save \
    --run_type train \
    --device 7
```

## Testing 
Testing on trained model:
```
  ./scripts/test.sh
```

Or:
```
  python tools/run.py \
    --config ./config/<vit5/bart>_config.yaml \
    --save_dir ./save \
    --run_type inference\
    --device 7
    --resume_file your/ckpt/path
```


