
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

# TODO

replicate the best baseline : visual bert pretrained from coco 
The config from facebook are [[here]](https://github.com/facebookresearch/mmf/tree/master/projects/visual_bert/configs/hateful_memes) <br/><br/>

We will replacate the from_coco.yaml setting, that includes the defaults.yaml settings, that itself includes the [[configs/datasets/hateful_memes/with_features.yaml setting]](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/datasets/hateful_memes/with_features.yaml). <br/><br/>

1) Créer une classe basé sur leur dataloader, ca doit etre [[celui-ci]](https://github.com/facebookresearch/mmf/blob/master/mmf/datasets/builders/hateful_memes/dataset.py#L17). Il faut aussi trouver les données du dataset et voir combien elles pèsent. Ici, j'ai l"impression qu'ils font les lecture disque dans le get_item, peut etre pcq c'est asssez gros. Je pense qu'une bonne idée, c'est d'utiliser leur framework pour construire les données, les récupérer, et puis simplifier le dataloader pour fonctionner direct sur les données (et se débarasser de tout le coté création de features que leur dataset propose).

2) créer une classe pour le modèle visual bert (instancié [[ici]](https://github.com/facebookresearch/mmf/blob/master/mmf/models/visual_bert.py#L384). Dans notre cas, c'est probablement VisualBERTForPretraining. L'idée, c'est probablement de lancer une fois leur projet, sortir les variable de config pour voir qu'est ce qui va où, copier le code chez nous et le modifier pour le faire marcher dans notre framework, c'est du boulot :) mais c'est ca aussi le challenge. 

3) repliquer la baseline avec les bonnes configs. 

