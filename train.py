import torch
import torch.nn as nn
import time
import itertools
import os
from torch.optim.lr_scheduler import LambdaLR
from utils.utils_func import _convert_to_one_hot
from sklearn.metrics import roc_auc_score

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(net, train_loader, eval_loader, weights, args):
    # best_eval_accuracy = 0.0
    # best_eval_auroc = 0.0
    # best_net = net.state_dict()
    # best_net_train_error = 0
    # early_stop = 0

    # Load optimizer, scheduler and loss function
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr_base, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optim, 2000, 22000)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Use batch_sampler associated with train_loader to keep track of indices
    train_batch_sampler = train_loader.batch_sampler

    eval_accuracies = []
    eval_auroc = []

    for epoch in range(0, args.max_epoch):

        time_start = time.time()
        loss_sum = 0
        net_train_error = 0

        for step, (samples_batch, indices_batch) in enumerate(zip(train_loader, train_batch_sampler)):

            loss_tmp = 0
            train_error_tmp = 0
            batch_size = len(indices_batch)
            targets = samples_batch["label"]

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
            weights_batch = weights[indices_batch]

            # Checking shape
            assert weights_batch.shape == torch.Size([batch_size])

            # Evaluate predictions from logits (threshold: 0.5) then the weighted misclassification error
            predictions = (torch.nn.Sigmoid(logits) > 0.5).long()
            assert predictions.shape == torch.Size([batch_size])
            assert predictions.dtype == torch.long
            incorrect = (predictions != targets).float()
            train_error = torch.dot(weights, incorrect)

            # Evaluate loss value for each sample in the batch - shape: (batch_size)
            loss = loss_fn(logits, targets)

            # Checking shape
            assert loss.shape == torch.Size([batch_size])

            # Weight the loss values then evaluate the mean
            loss = torch.mean(torch.mul(weights, loss))

            # Backward pass
            loss.backward()

            loss_tmp += loss.cpu().data.numpy()
            train_error_tmp += train_error.cpu().data.numpy()

            loss_sum += loss.cpu().data.numpy()
            net_train_error += train_error.cpu().data.numpy()

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
                nn.utils.clip_grad_norm_(
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
        print('Weighted train loss: ', loss_sum)
        print('Weighted train misclassification error: ', net_train_error)
        # epoch_finish = epoch + 1


        # # Eval
        # if epoch_finish >= args.eval_start:
        #     print('Evaluation...')
        #     accuracy, auroc = evaluate(net, eval_loader, args)
        #     print('Accuracy :' + str(accuracy))
        #     print('Auroc :' + str(auroc))

        #     eval_accuracies.append(accuracy)
        #     eval_auroc.append(auroc)

        #     if accuracy > best_eval_accuracy:
        #         # Best net
        #         state = {
        #             'state_dict': net.state_dict(),
        #             'optimizer': optim.state_dict(),
        #             'args': args,
        #         }
        #         torch.save(
        #             state,
        #             args.output + "/" + args.name +
        #             '/best' + str(args.seed) + '.pkl'
        #         )
        #         best_eval_accuracy = accuracy
        #         best_eval_auroc = auroc
        #         best_net = net.state_dict()
        #         best_net_train_error = net_train_error

        #         early_stop = 0

        #     else:
        #         # Early stop
        #         early_stop += 1
        #         if early_stop == args.early_stop:
        #             print('Early stop reached')
        #             print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
        #             os.rename(args.output + "/" + args.name +
        #                       '/best' + str(args.seed) + '.pkl',
        #                       args.output + "/" + args.name +
        #                       '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
        #             return eval_accuracies, eval_auroc, best_net, best_net_train_error

        # loss_sum = 0

        net_state = net.state_dict()

    return net_state, loss_sum, net_train_error 


def evaluate(net, eval_loader, args):
    with torch.no_grad():
        net.eval()
        acc_correct = 0
        acc_total = 0
        roc_expected = torch.tensor([])
        roc_output = torch.tensor([])

        for step, samples_batch in enumerate(eval_loader):

            for key, value in samples_batch.items():
                samples_batch[key] = value.to('cuda')
            out = net(samples_batch)

            output = out["scores"]
            expected = samples_batch["label"]

            # Accuracy
            if output.dim() == 2:
                output = torch.max(output, 1)[1]
            # If last dim is 1, we directly have class indices
            if expected.dim() == 2 and expected.size(-1) != 1:
                expected = torch.max(expected, 1)[1]
            acc_correct += (expected == output.squeeze()).sum().float()
            acc_total += len(expected)

            # Auroc
            output = torch.nn.functional.softmax(out["scores"], dim=-1)
            expected = samples_batch["label"]
            expected = _convert_to_one_hot(expected, output)
            roc_expected = torch.cat((roc_expected, expected.cpu()))
            roc_output = torch.cat((roc_output, output.cpu()))

        # acc
        acc = acc_correct / acc_total

        # auroc
        auroc = roc_auc_score(roc_expected, roc_output)
        auroc = expected.new_tensor(auroc, dtype=torch.float)

        print(acc, auroc)
        return acc, auroc

