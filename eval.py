import torch
from torch import nn
import numpy as np 

from models.visual_bert import PrepareVisualBertModel


def evaluate_visual_bert(eval_loader, args):
    model = PrepareVisualBertModel(args)
    model.train(False)
    
    preds = []
    probs = []
    pred_labels = []
    true_labels = []
    accs = []
    total_correct = 0
    total_samples = 0

    for _ , samples_batch in enumerate(eval_loader):

        # samples_batch['input_ids'] = samples_batch['input_ids'].cuda()
        # samples_batch['segment_ids'] = samples_batch['segment_ids'].cuda()
        # samples_batch['input_mask'] = samples_batch['input_mask'].cuda()
        # samples_batch['img_features'] = samples_batch['img_features'].cuda()

        pred = model(samples_batch)["scores"]
        prob = nn.Softmax(dim=1)(pred)
        
        pred = pred.detach().numpy()
        prob = prob.detach().numpy()
        pred_label = np.argmin(prob, axis=1)
        true_label = samples_batch['label'].detach().numpy()
        correct = np.sum(pred_label == true_label)
        samples = samples_batch['input_ids'].shape[0]
        acc = correct / samples

        preds.append(pred)
        probs.append(prob)
        pred_labels.append(pred_label)
        true_labels.append(true_label)
        accs.append(acc)

        total_correct += correct
        total_samples += samples
    
    total_acc = total_correct / total_samples

    return preds, probs, pred_labels, true_labels, accs, total_acc

        # if not eval_loader.dataset.name == "test":
        #     true_label = true_label.cpu().data.numpy()
        #     accuracy += list((pred > 0) == true_label)

    # return 100 * np.mean(np.array(accuracy)), preds