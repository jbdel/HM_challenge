import torch
from torch import nn
import numpy as np 

from models.visual_bert import PrepareVisualBertModel


def evaluate_visual_bert(eval_loader, args):
    model = PrepareVisualBertModel(args).cuda()
    model.train(False)
    # accuracy = []
    preds = []
    probs = []
    true_labels = []
    for _ , samples_batch in enumerate(eval_loader):

        samples_batch['input_ids'] = samples_batch['input_ids'].cuda()
        samples_batch['segment_ids'] = samples_batch['segment_ids'].cuda()
        samples_batch['input_mask'] = samples_batch['input_mask'].cuda()
        samples_batch['img_features'] = samples_batch['img_features'].cuda()

        pred = model(samples_batch).cpu().data

        softmax = nn.Softmax(dim=1)

        prob = softmax(pred)
        
        pred = pred.numpy()

        prob = prob.numpy()

        preds.append(pred)

        probs.append(prob)

        true_labels.append(samples_batch['label'].data.numpy())

    return preds, probs, true_labels

        # if not eval_loader.dataset.name == "test":
        #     true_label = true_label.cpu().data.numpy()
        #     accuracy += list((pred > 0) == true_label)

    # return 100 * np.mean(np.array(accuracy)), preds