import torch
import torch.nn as nn
import time
import numpy as np
import os


def train(net, train_loader, eval_loader, args):
    loss_sum = 0
    best_eval_accuracy = 0.0
    early_stop = 0

    # Load the optimizer parameters
    optim = torch.optim.Adam(net.parameters(), lr=args.lr_base)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_accuracies = []

    for epoch in range(0, args.max_epoch):

        time_start = time.time()

        for step, sample in enumerate(train_loader):

            loss_tmp = 0
            optim.zero_grad()

            img = sample['img'].cuda()
            text = sample['text'] # todo
            ans = sample['label'].cuda()

            pred = net(img, text)
            loss = loss_fn(pred, ans)
            loss.backward()

            loss_sum += loss.cpu().data.numpy()
            loss_tmp += loss.cpu().data.numpy()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m "
                  "remaining" % (
                      epoch + 1,
                      step,
                      int(len(train_loader.dataset) / args.batch_size),
                      loss_tmp / args.batch_size,
                      *[group['lr'] for group in optim.param_groups],
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

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1


        # Eval
        if epoch_finish >= args.eval_start:
            print('Evaluation...')
            accuracy, _ = evaluate(net, eval_loader, args)
            print('Accuracy :' + str(accuracy))
            eval_accuracies.append(accuracy)

            if accuracy > best_eval_accuracy:
                # Best
                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.state_dict(),
                    'args': args,
                }
                torch.save(
                    state,
                    args.output + "/" + args.name +
                    '/best' + str(args.seed) + '.pkl'
                )
                best_eval_accuracy = accuracy
                early_stop = 0

            else:
                # Early stop
                early_stop += 1
                if early_stop == args.early_stop:
                    print('Early stop reached')
                    print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
                    os.rename(args.output + "/" + args.name +
                              '/best' + str(args.seed) + '.pkl',
                              args.output + "/" + args.name +
                              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
                    return eval_accuracies

        loss_sum = 0


def evaluate(net, eval_loader, args):
    accuracy = []
    net.train(False)
    preds = {}
    for step, sample in enumerate(eval_loader):

        img = sample['img'].cuda()
        text = sample['text']
        ans = sample['label'].cuda()

        pred = net(img, text).cpu().data.numpy()

        if not eval_loader.dataset.name == "test":
            ans = ans.cpu().data.numpy()
            accuracy += list((pred > 0) == ans)


    net.train(True)
    return 100 * np.mean(np.array(accuracy)), preds

