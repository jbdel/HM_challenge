import torch
import numpy as np 

from train import train


def adaboost(base_net, n_estimators, train_loader, eval_loader, args):
    " AdaBoost for weight boosting and ensemble learning."

    n_train_samples = len(train_loader.dataset)
    n_eval_samples = len(eval_loader.dataset)

    samples_weights = torch.ones(n_train_samples, dtype=torch.float64) / n_train_samples

    estimators = []
    estimators_weights = torch.zeros(n_estimators, dtype=torch.float64)
    estimators_errors = torch.ones(n_estimators, dtype=torch.float64)

    n_epochs = args.max_epoch

    for step in range(n_estimators):
        # Training step using weighted samples and weighted loss function over n_epochs
        # Choose best model over n_epochs with respect to accuracy on the eval dataset
        net_state, net_loss_sum, net_train_error = train(base_net, 
                                                     train_loader, 
                                                     eval_loader, 
                                                     samples_weights, args)
        # Stop if classification is perfect
        if net_train_error == 0:
            return None, None, None
        
        # Stop if error is worst than random guessing 
        if net_train_error >= 0.5:
            if len(estimators) == 0:
                raise ValueError('Base Estimator worse than random, ensemble learning cannot be fit')
            else: 
                print('Estimator at this step is worse than random, stop iteration here')

        # Keep estimator at this step in memory
        estimators.append(net_state)

        # Evaluate estimator weight
        estimator_weight = 





