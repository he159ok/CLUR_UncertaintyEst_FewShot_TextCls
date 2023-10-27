import torch
import numpy as np

from scipy.special import softmax

from operator import itemgetter

import sklearn

from torch.autograd import Variable

def attention_masks(input_ids, src_pad_idx=0):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [int(i != src_pad_idx) for i in seq]
        atten_masks.append(seq_mask)
    atten_masks = torch.tensor(atten_masks)
    return atten_masks



def cal_random_norm_aug_loss(repeat_norm_ranges, y_pred_tensor, y):
    repeat_times = len(repeat_norm_ranges)
    batch_size, dim_num = y_pred_tensor.shape
    mid_vec = torch.zeros(dim_num) * 1.0
    mid_vec[0] = 1.0
    mid_val = torch.nn.LogSoftmax(dim=0)(mid_vec)[0]
    target = torch.ones(batch_size).cuda() * mid_val
    y_pre_logsoftmax = torch.nn.LogSoftmax(dim=1)(y_pred_tensor)

    outputs = y_pre_logsoftmax[range(batch_size), y]
    input_dif = outputs - target

    index = np.random.randint(0, repeat_times, size=1)[0]

    mid_loss = torch.norm(input_dif, p=repeat_norm_ranges[index])
    return mid_loss



def cross_entory_aug(y_pre, y, sample_weight, aug_mode=1):
    # https: // stackoverflow.com / questions / 58360150 / how - to - write - custom - crossentropyloss
    batch_size = y.shape[0]
    sample_weight_orisum = torch.sum(sample_weight)
    sample_weight = sample_weight * (batch_size/sample_weight_orisum)
    if aug_mode == 2:
        y_pre_std = torch.nn.Dropout(0.2)(y_pre)
        eps = torch.randn_like(y_pre_std) * 0.05
        y_pre = y_pre + y_pre_std * eps
    y_pre_logsoftmax = torch.nn.LogSoftmax(dim=1)(y_pre)
    outputs = y_pre_logsoftmax[range(batch_size), y]
    if aug_mode == 1:
        weighted_outputs = outputs * sample_weight.cuda()
    elif aug_mode == 2:
        weighted_outputs = outputs

    return -torch.sum(weighted_outputs)/batch_size


def weighted_aux_cross_entory_loss(y_pre, y):
    # https: // stackoverflow.com / questions / 58360150 / how - to - write - custom - crossentropyloss
    batch_size = y.shape[0]
    sample_weight = calc_entropy(y_pre)

    y_pre_logsoftmax = torch.nn.LogSoftmax(dim=1)(y_pre)
    outputs = y_pre_logsoftmax[range(batch_size), y]

    weighted_outputs = outputs * sample_weight.cuda()
    return -torch.sum(weighted_outputs)/batch_size



def seq_cross_entory_aug(y_pre, y, sample_weight, aug_mode = 1):
    # https: // stackoverflow.com / questions / 58360150 / how - to - write - custom - crossentropyloss
    batch_size, dim_num = y_pre.shape
    mid_vec = torch.zeros(dim_num) * 1.0
    mid_vec[0] = 1.0
    mid_val = torch.nn.LogSoftmax(dim=0)(mid_vec)[0]
    target = torch.ones(batch_size).cuda() * mid_val
    sample_weight_orisum = torch.sum(sample_weight)
    sample_weight = sample_weight * (batch_size/sample_weight_orisum)
    sample_weight = sample_weight.cuda()
    if aug_mode == 2: # add noise to the loss function
        y_pre_std = torch.nn.Dropout(0.2)(y_pre)
        eps = torch.randn_like(y_pre_std) * 0.05
        y_pre = y_pre + y_pre_std * eps
    y_pre_logsoftmax = torch.nn.LogSoftmax(dim=1)(y_pre)
    outputs = y_pre_logsoftmax[range(batch_size), y]
    if aug_mode == 1:
        weighted_outputs = outputs * sample_weight
    elif aug_mode == 2:
        weighted_outputs = outputs


    weighted_target = target * sample_weight
    mid_loss = torch.nn.MSELoss()(weighted_outputs, weighted_target)

    return torch.sum(mid_loss)/batch_size



def human_idk_un_metric(un_score_list, y_truth, y_pred, dataset):

    indices, L_sorted = zip(*sorted(enumerate(un_score_list), key=itemgetter(1), reverse=False))
    idk_list = np.arange(0, 0.5, 0.1)
    for idk_ratio in idk_list:
        # print("=== idk_ratio: ", idk_ratio, " ===")
        test_num = int(len(L_sorted) * (1 - idk_ratio))
        indices_cur = list(indices[:test_num])
        y_truth_cur = [y_truth[i] for i in indices_cur]
        y_pred_cur = [y_pred[i] for i in indices_cur]

        human_indices = list(indices[test_num:])
        y_human = [y_truth[i] for i in human_indices]
        y_truth_cur = y_truth_cur + y_human
        y_pred_cur = y_pred_cur + y_human

        f1_score = show_results(dataset, y_truth_cur, y_pred_cur)


def idk_un_metric(un_score_list, y_truth, y_pred, dataset):

    indices, L_sorted = zip(*sorted(enumerate(un_score_list), key=itemgetter(1), reverse=True))

    idk_list = np.arange(0, 0.5, 0.1)
    for idk_ratio in idk_list:
        test_num = int(len(L_sorted) * (1 - idk_ratio))
        indices_cur = list(indices[:test_num])
        y_truth_cur = [y_truth[i] for i in indices_cur]
        y_pred_cur = [y_pred[i] for i in indices_cur]
        f1_score = show_results(dataset, y_truth_cur, y_pred_cur)

def show_results(dataset, y_truth, y_pred):

    class_num = dataset.get_class_num()

    if class_num == 2:
        accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
        f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1))
        prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, pos_label=1))
        recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, pos_label=1))


        print(accuracy_score, f1_score, prec_score, recall_score, sep='\t')

        return f1_score

    elif class_num > 2:
        accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
        micro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='micro'))
        macro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='macro'))
        print(accuracy_score, micro_f1_score, macro_f1_score, sep='\t')

        return micro_f1_score


def calc_entropy(input_tensor):
    lsm = torch.nn.LogSoftmax(dim=1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -1 * p_log_p.mean(dim=1)
    return entropy
