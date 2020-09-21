import torch
import numpy as np 
import time
import copy
import itertools
import os
from torch.optim.lr_scheduler import LambdaLR
# from utils.utils_func import _convert_to_one_hot
# from sklearn.metrics import roc_auc_score

def adaboost(base_estimator, n_estimators, train_loader, eval_loader, args):
    " AdaBoost for weight boosting and ensemble learning."

    n_train_samples = len(train_loader.dataset)
    n_eval_samples = len(eval_loader.dataset)
    eval_targets = get_eval_targets(eval_loader, n_eval_samples)

    assert eval_targets.shape == torch.Size([n_eval_samples])
    assert eval_targets.dtype == torch.long 
    print(eval_targets)

    samples_weights = torch.ones(n_train_samples, dtype=torch.float64) / n_train_samples

    estimators = []
    estimator_weights = torch.zeros(n_estimators, dtype=torch.float64)
    estimator_errors = torch.ones(n_estimators, dtype=torch.float64)
    estimator_preds = []

    ensemble_accuracies = []

    for step in range(n_estimators):

        # Check that weights are positive and sum to one
        assert torch.isclose(torch.sum(samples_weights), torch.zeros(1, dtype=torch.float64))
        assert torch.min(samples_weights) >= 0

        # Deepcopy base_estimator
        estimator = copy.deepcopy(base_estimator)

        # Training step using weighted samples and weighted loss function over n_epochs
        estimator_state, _, estimator_train_error, estimator_incorrect = ada_train(
                                                    estimator, 
                                                    train_loader,  
                                                    samples_weights,
                                                    args
                                                )

        # Check estimator_incorrect shape and type
        assert estimator_incorrect.shape == torch.Size([n_train_samples])
        assert estimator_incorrect.dtype == torch.float64

        # Stop if classification is perfect
        if estimator_train_error == 0:
            raise ValueError('Perfect training classification achieved')
        
        # Stop if error is worst than random guessing 
        if estimator_train_error >= 0.5:
            if len(estimators) == 0:
                raise ValueError('Base Estimator worse than random, ensemble learning cannot be fit')
            else: 
                raise ValueError('Estimator at this step is worse than random, stop iteration here')

        # Keep estimator state and train error at this step in memory
        estimators.append(estimator_state)
        estimator_errors[step] = estimator_train_error

        # Evaluate estimator weight and keep it in memory
        estimator_weight = np.log((1. - estimator_train_error) / estimator_train_error)
        estimator_weights[step] = estimator_weight

        # Update sample weights according to estimator_weight and estimator_incorrect if not last step
        if not step == n_estimators - 1:
            samples_weights *= np.exp(estimator_weight * estimator_incorrect)
            samples_weights /= torch.sum(samples_weights)
        
        # Evaluate the predictions of the current estimator on the eval_loader
        estimator_pred = estimator_eval(estimator, eval_loader, n_eval_samples)
        estimator_preds.append(estimator_pred)

        # Evaluate accuracy (and auroc) for ensemble learning at this point
        ensemble_accuracy = ensemble_eval(estimator_preds, estimator_weights, eval_targets)
        print(ensemble_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)

    return estimators, estimator_weights, estimator_errors, ensemble_accs

def get_eval_targets(eval_loader, n_eval_samples):

    eval_targets = torch.zeros(n_eval_samples, dtype=torch.long)
    start = 0

    for samples_batch in eval_loader:
        batch_size = len(samples_batch)
        eval_targets[start:start + batch_size] = samples_batch["label"].long()
        start = start + batch_size
        
    return eval_targets

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def ada_train(net, train_loader, samples_weights, args):

    # Load optimizer, scheduler and loss function
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr_base, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optim, 2000, 22000)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Use batch_sampler associated with train_loader to keep track of indices
    train_batch_sampler = train_loader.batch_sampler

    for epoch in range(args.max_epoch):

        time_start = time.time()
        net_loss_sum = 0
        net_train_error = 0
        net_incorrect = torch.ones(samples_weights.shape, dtype=torch.float64)

        for step, (samples_batch, indices_batch) in enumerate(zip(train_loader, train_batch_sampler)):

            loss_tmp = 0
            train_error_tmp = 0
            batch_size = len(indices_batch)
            targets = samples_batch["label"].long()

            # Checking shape and dtype
            assert targets.shape == torch.Size([batch_size])
            assert targets.dtype == torch.long

            # Set gradients to zero before forward pass
            optim.zero_grad()

            # Send inputs to CUDA
            for key, value in samples_batch.items():
                samples_batch[key] = value.to('cuda')

            # Evaluate logits predictions - shape: (batch_size)
            output_dic = net(samples_batch)
            logits = output_dic["scores"]

            # Checking shape
            assert logits.shape == torch.Size([batch_size])

            # Pick corresponding weights for batch samples - shape: (batch_size)
            weights_batch = samples_weights[indices_batch]

            # Checking shape
            assert weights_batch.shape == torch.Size([batch_size])

            # Evaluate predictions from logits (threshold: 0.5) then the weighted misclassification error
            predictions = (torch.nn.Sigmoid(logits) > 0.5).long()
            assert predictions.shape == torch.Size([batch_size])
            assert predictions.dtype == torch.long
            incorrect = (predictions != targets).double()
            train_error = torch.dot(weights_batch, incorrect)

            # Evaluate loss value for each sample in the batch - shape: (batch_size)
            loss = loss_fn(logits, targets)

            # Checking shape
            assert loss.shape == torch.Size([batch_size])

            # Weight the loss values then evaluate the mean
            loss = torch.mean(torch.mul(weights_batch, loss))

            # Backward pass
            loss.backward()

            loss_tmp += loss.cpu().data.numpy()
            train_error_tmp += train_error.cpu().data.numpy()

            net_loss_sum += loss.cpu().data.numpy()
            net_train_error += train_error.cpu().data.numpy()
            net_incorrect[indices_batch] = incorrect.cpu().data.numpy()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Error: %.4f, Lr: %.2e, %4d m "
                  "remaining" % (
                      epoch + 1,
                      step,
                      int(len(train_loader.dataset) / args.batch_size),
                      loss_tmp / args.batch_size,
                      train_error_tmp,
                      *[group['lr'] for group in scheduler.optimizer.param_groups],
                      ((time.time() - time_start) / (step + 1)) * (
                                  (len(train_loader.dataset) / args.batch_size) - step) / 60,
                  ), end='          ')

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )
            optim.step()
            # schedule every batch or every epoch ?
            # every batch : cf https://github.com/facebookresearch/mmf/blob/48631d8ec7d73fcd5f8cb24582e907525c1691c0/mmf/trainers/callbacks/lr_scheduler.py#L24
            scheduler.step()

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        print('Weighted train loss: ', net_loss_sum)
        print('Weighted train misclassification error: ', net_train_error)

        net_state = net.state_dict()

    return net_state, net_loss_sum, net_train_error, net_incorrect

def estimator_eval(estimator, eval_loader, n_eval_samples):

    estimator_pred = torch.zeros(n_eval_samples, dtype=torch.long)

    with torch.no_grad():
        estimator.eval()
        start = 0
    
        for step, samples_batch in enumerate(eval_loader):

            for key, value in samples_batch.items():
                samples_batch[key] = value.to('cuda')

            output_dic = estimator(samples_batch)
            logits = output_dic["scores"]
            predictions = (torch.nn.Sigmoid(logits) > 0.5).long()

            # Outputs -1 or +1 depending on the prediction 0 or 1
            predictions = (predictions * 2) - 1

            batch_size = len(predictions)
            estimator_pred[start:start + batch_size] = predictions
            start = start + batch_size
        
        return estimator_pred


def ensemble_eval(estimator_preds, estimator_weights, eval_targets):

    ensemble_pred = torch.zeros(len(eval_targets), dtype=torch.double)

    assert len(estimator_preds) == len(estimator_weights)

    for n, estimator_pred in enumerate(estimator_preds):
        ensemble_pred += estimator_weights[n] * estimator_pred
    
    # Obtain predictions with targets 0 and 1
    ensemble_pred = ((torch.sign(ensemble_pred) + 1.) * 0.5).long()

    # Accuracy
    ensemble_accuracy = torch.mean((ensemble_pred == eval_targets).double())

    # AUROC: TO DO

    return ensemble_accuracy
