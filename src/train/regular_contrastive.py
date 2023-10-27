import os
import time
import datetime

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from train.utils import named_grad_param, grad_param, get_norm

import torch.nn.functional as F

from train.uneq_contra_loss import UnEqualContraLoss, ContraLoss

def train(train_data, val_data, model, args):
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'max', patience=args.patience//2, factor=0.1, verbose=True)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    train_gen = ParallelSampler(train_data, args, args.train_episodes)
    train_gen_val = ParallelSampler(train_data, args, args.val_episodes)
    val_gen = ParallelSampler(val_data, args, args.val_episodes)

    for ep in range(args.train_epochs):
        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'ebd': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad)

        if ep % 10 == 0:
            acc, std = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("train stats", "cyan"),
               colored("ebd_grad:", "blue"), np.mean(np.array(grad['ebd'])),
               colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
               ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.ebd')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
            flush=True)

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            best_path), flush=True)

        torch.save(model['ebd'].state_dict(), best_path + '.ebd')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    model['clf'].train()
    opt.zero_grad()

    support, query = task

    # Embedding the document
    XS1, TS_onehot1, alphaS1, XS2, TS_onehot2, alphaS2 = model['ebd'](support)
    YS = support['label']

    XQ1, TQ_onehot1, alphaQ1, XQ2, TQ_onehot2, alphaQ2, = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    if args.modelmode == "clur":
        C_YS, C_YQ, Q_pred1, Q_pred2, Q_logit1, Q_logit2 = model['clf'](XS1, XS2, YS, XQ1, XQ2, YQ, mixup_y = TS_onehot1, mixup_y2 = TS_onehot2)

    if args.contrastive and args.modelmode == "clur":
        if args.feature_aug_mode != None and (args.use_unequal or args.use_equal):
            ini_loss_cr1 = F.cross_entropy(Q_logit1, C_YQ, reduction='none') - (-1) * torch.log(
                torch.FloatTensor([args.unequal_lower_boundary])).cuda()
            ini_loss_cr2 = F.cross_entropy(Q_logit2, C_YQ, reduction='none') - (-1) * torch.log(
                torch.FloatTensor([args.unequal_lower_boundary])).cuda()
            loss_cr1 = torch.mean(torch.clamp(ini_loss_cr1, min=0))
            loss_cr2 = torch.mean(torch.clamp(ini_loss_cr2, min=0))

        # below is contrastive loss
        loss_contrastive = 0
        if args.feature_aug_mode != None and args.use_unequal:
            if args.unequal_type == 0:
                uneq_contra_loss = UnEqualContraLoss()
                loss_contrastive_1 = 0.5 * uneq_contra_loss(Q_logit1, Q_pred2, TQ_onehot1,
                                                            TQ_onehot2)
                loss_contrastive_2 = 0.5 * uneq_contra_loss(Q_logit2, Q_pred1, TQ_onehot2,
                                                            TQ_onehot1)
                loss_contrastive = loss_contrastive_1 + loss_contrastive_2
            elif args.unequal_type == 1:
                uneq_contra_loss = UnEqualContraLoss()
                loss_contrastive_1 = 0.5 * uneq_contra_loss(Q_logit1, Q_logit2, TQ_onehot1,
                                                            TQ_onehot2)
                loss_contrastive_2 = 0.5 * uneq_contra_loss(Q_logit2, Q_logit1, TQ_onehot2,
                                                            TQ_onehot1)
                loss_contrastive = loss_contrastive_1 + loss_contrastive_2
            elif args.unequal_type == 2:
                uneq_contra_loss = UnEqualContraLoss()
                loss_contrastive_1 =  uneq_contra_loss(Q_logit1, Q_logit2, TQ_onehot1,
                                                            TQ_onehot2, is_detach=False)
                loss_contrastive = loss_contrastive_1
        elif args.feature_aug_mode != None and args.use_equal:
            equal_contra_loss = ContraLoss()
            loss_contrastive_1 = 0.5 * equal_contra_loss(Q_logit1, Q_pred2)
            loss_contrastive_2 = 0.5 * equal_contra_loss(Q_logit2, Q_pred1)
            loss_contrastive = loss_contrastive_1 + loss_contrastive_2


    if args.modelmode == 'clur':
        loss = loss_cr1 + loss_cr2 + loss_contrastive * args.contrast_weight


    if loss is not None:
        loss.backward()

    if torch.isnan(loss):
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']),
                                  args.clip_grad)

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()

    acc = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc.append(test_one(task, model, args))

    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                ), flush=True)

    return np.mean(acc), np.std(acc)


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task

    # Embedding the document
    XS1, TS_onehot1, alphaS1, XS2, TS_onehot2, alphaS2 = model['ebd'](support)
    YS = support['label']

    XQ1, TQ_onehot1, alphaQ1, XQ2, TQ_onehot2, alphaQ2 = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    if args.modelmode == "clur":
        C_YS, C_YQ, Q_pred1, Q_pred2, Q_logit1, Q_logit2 = model['clf'](XS1, XS2, YS, XQ1, XQ2, YQ, mixup_y = TS_onehot1, mixup_y2 = TS_onehot2)

    acc1 = torch.mean((torch.argmax(Q_logit1, dim=1) == C_YQ).float()).item()

    return acc1
