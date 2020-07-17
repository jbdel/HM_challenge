
Google Colab sheet: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1in1Pmj7TL4kGBXlO9iSKe8x6B5D_4mwl?usp=sharing)
# HM_challenge

```
.
├── HMDataset
│   ├── data
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   ├── train.jsonl
├── HMDataset.py
├── main.py
├── models
│   ├── __init__.py
│   └── resnet.py
├── train.py
└── utils
    └── utils_loader.py
```

 [[Download data]](https://www.dropbox.com/s/dy0ugzx7m7dl5c2/img_reduced.zip?dl=1) and place the unzipped folder in `HMDataet/data/`

```
python main.py --model Model_Resnet --lr_base 0.001 --batch_size 32
```

To create a new model, create new class in models folder, init must take one argument (args) and forward takes two arguments (img and text). Declare created class in `__init__.py` file and call it through `--model`argument of `main.py`
