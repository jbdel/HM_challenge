import torch
from torch import nn
import numpy as np 

from models.visual_bert import PrepareVisualBertModel


def evaluate_visual_bert(eval_loader):
    model = PrepareVisualBertModel().cuda()
    model.train(False)
    # accuracy = []
    preds = []
    for _ , samples_batch in enumerate(eval_loader):

        input_ids = samples_batch['input_ids'].cuda()
        segment_ids = samples_batch['segment_ids'].cuda()
        input_mask = samples_batch['input_mask'].cuda()
        img_features = samples_batch['img_features'].cuda()
        true_label = samples_batch['label'].cuda()

        pred = model(input_ids, segment_ids, img_features, input_mask).cpu().data.numpy()

        preds.append(pred)

    return preds

        # if not eval_loader.dataset.name == "test":
        #     true_label = true_label.cpu().data.numpy()
        #     accuracy += list((pred > 0) == true_label)

    # return 100 * np.mean(np.array(accuracy)), preds