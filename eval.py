import torch
from torch import nn
import numpy as np 

from models.visual_bert import PrepareVisualBertModel


def evaluate_visual_bert(eval_loader, args):
    model = PrepareVisualBertModel(args).cuda()
    model.train(False)
    # accuracy = []
    preds = []
    for _ , samples_batch in enumerate(eval_loader):

        samples_batch['input_ids'] = samples_batch['input_ids'].cuda()
        samples_batch['segment_ids'] = samples_batch['segment_ids'].cuda()
        samples_batch['input_mask'] = samples_batch['input_mask'].cuda()
        samples_batch['img_features'] = samples_batch['img_features'].cuda()
        samples_batch['label'] = samples_batch['label'].cuda()

        pred = model(samples_batch).cpu().data.numpy()

        preds.append(pred)

    return preds

        # if not eval_loader.dataset.name == "test":
        #     true_label = true_label.cpu().data.numpy()
        #     accuracy += list((pred > 0) == true_label)

    # return 100 * np.mean(np.array(accuracy)), preds