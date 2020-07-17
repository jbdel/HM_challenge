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

Download data from [[here]](https://www.dropbox.com/s/9h3qdmj80gc6pbl/img_reduced.zip?dl=1) and place the unzipped folder in `HMDataet/data/`

```
python main.py --model Model_Resnet --lr_base 0.001 --batch_size 32
```

To create a new model, create new class in models folder, init must take one argument (args) and forward takes two arguments (img and text). Declare created class in `__init__.py` file and call it through `--model`argument of `main.py`
