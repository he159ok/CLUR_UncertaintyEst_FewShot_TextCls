# This file is revised from drop_entropy_eval_contrastive.py


from operator import itemgetter


import torch

import numpy as np

import scipy
import scipy.stats



import torch.nn as nn


from dataset.parallel_sampler import ParallelSampler
from tqdm import tqdm
from termcolor import colored

import sklearn
import sklearn.metrics

def show_roc_auc(y_truth, y_pred_softmax, unc_list, cal_aupr = True):  # 应该百分比

	y_truth = np.array(y_truth)
	y_pred_label = torch.argmax(y_pred_softmax, dim=1).numpy()
	y_binary = (y_pred_label == y_truth).astype(int)
	conf_array = 1.0 / (np.array(unc_list)+1.0)
	fpr, tpr, thresholds_auroc = sklearn.metrics.roc_curve(y_binary, conf_array, pos_label=1)
	auroc_score = sklearn.metrics.auc(fpr, tpr)

	precision, recall, thresholds_aupr, aupr_score = None, None, None, None

	if cal_aupr:
		precision, recall, thresholds_aupr = sklearn.metrics.precision_recall_curve(y_binary, conf_array, pos_label=1)
		aupr_score = sklearn.metrics.auc(recall, precision)

	return auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score



def show_results(class_num_flag, y_truth, y_pred, represent=None, target=None, writef_name=None):


	if class_num_flag == 1:
		accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
		f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, pos_label=1))
		prec_score = (sklearn.metrics.precision_score(y_truth, y_pred, pos_label=1))
		recall_score = (sklearn.metrics.recall_score(y_truth, y_pred, pos_label=1))
		confusion_mat = (sklearn.metrics.confusion_matrix(y_truth, y_pred))


		return f1_score

	elif class_num_flag == 0:
		accuracy_score = (sklearn.metrics.accuracy_score(y_truth, y_pred))
		micro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='micro'))
		macro_f1_score = (sklearn.metrics.f1_score(y_truth, y_pred, average='macro'))


		return macro_f1_score 




def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax(dim=1)
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean(dim=1)
    return entropy

def cal_unequal_score(p, z, alpha1, alpha2):
    target_onehot_processed1 = 1 - alpha1
    target_onehot_processed2 = 1 - alpha2
    sign_index = (target_onehot_processed1 - target_onehot_processed2)


    p_Fnorm = calc_entropy(p)
    z_Fnorm = calc_entropy(z)

    ini_loss = -1 * (z_Fnorm - p_Fnorm)
    ini_loss2 = ini_loss * sign_index.squeeze(1)
    unequal_score = torch.clamp(ini_loss2, min=0)
    return unequal_score

def logit_score(logit, top_k):
    score_list = []
    for idx in range(len(logit)):
        logit_i = logit[idx].data.cpu().numpy().tolist()
        indices, L_sorted = zip(*sorted(enumerate(logit_i), key=itemgetter(1), reverse=True))
        score_i = (top_k * L_sorted[0] - sum(L_sorted[1:top_k + 1])) / top_k
        score_list.append(score_i)
    return score_list

def drop_freq(y_probs):
    y_probs = np.array(y_probs)
    n = y_probs.shape[0]
    drop_score_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        counts = np.bincount(logit_i)
        max_count = np.max(counts)
        drop_score_list.append(max_count / n)
    return drop_score_list

def drop_entropy(y_prbos, mask_num):
    y_probs = np.array(y_prbos)
    entropy_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        entropy_list.append(entropy)
    return entropy_list

def drop_entropy_naacl(y_prbos, mask_num, logit_sfmean, logit_sfvar, represent_setvar):
    y_probs = np.array(y_prbos)
    entropy_list = []
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        entropy_list.append(entropy)


    return entropy_list

def standard_data(data):
    data = np.array(data)
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    res = (data - mu) * 1.0 / sigma
    res = list(res)
    return res


def normalize_data(data):
    data = np.array(data)
    min_value = np.min(data)
    range0 = np.max(data) - min_value
    res = (data - min_value) * 1.0 / range0
    res = list(res)
    return res

def drop_entropy_emnlp(y_prbos, mask_num, logit_sfmean, logit_sfvar, represent_setvar):
    y_probs = np.array(y_prbos)
    entropy_list = []
    y_inner_confidence = np.max(logit_sfmean, axis = 1).tolist()
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        inner_betw_weighted_entropy = 1/y_inner_confidence[i]
        entropy_list.append(inner_betw_weighted_entropy)
    return entropy_list

def drop_entropy_emnlp2(y_prbos, mask_num, logit_sfmean, logit_sfvar, represent_setvar):
    y_probs = np.array(y_prbos)
    entropy_list = []
    entropy_list2 = []
    y_inner_confidence = np.max(logit_sfmean, axis = 1).tolist()
    for i in range(y_probs.shape[1]):
        logit_i = y_probs[:, i]
        bin_count = np.bincount(logit_i)
        mask = sorted(range(len(bin_count)), key=lambda i: bin_count[i])[:-mask_num]
        bin_count[mask] = 0
        count_probs = [i / sum(bin_count) for i in bin_count]

        entropy = scipy.stats.entropy(count_probs)
        inner_betw_weighted_entropy = 1 / y_inner_confidence[i]
        entropy_list.append(entropy)
        entropy_list2.append(inner_betw_weighted_entropy)
    return entropy_list, entropy_list2

def uncertain_score(support, query, model, drop_num = 20, mask_num = 5, use_sec_unc = False, para=None, use_fea_diff = False):
    # dropout
    y_probs = []

    logit_sfset = []
    represent_set = []
    unc_score_set = []

    assert use_fea_diff == False

    for i in range(1):
        # Embedding the document
        XS1, TS_onehot1, alphaS1, XS2, TS_onehot2, alphaS2 = model['ebd'](support, use_fea_diff=use_fea_diff)
        YS = support['label']

        XQ1, TQ_onehot1, alphaQ1, XQ2, TQ_onehot2, alphaQ2 = model['ebd'](query, use_fea_diff=use_fea_diff)
        YQ = query['label']
        # Apply the classifier
        res = model['clf'](XS1, XS2, YS, XQ1, XQ2, YQ)

        C_YS, C_YQ, Q_pred1, Q_pred2, Q_logit1, Q_logit2 = res[0], res[1], res[2], res[3], res[4], res[5]

        logit, represent, logit2, represent2, alpha, alpha2 = Q_logit1, Q_pred1, Q_logit2, Q_pred2, alphaQ1, alphaQ2
        logit = torch.nn.Softmax(dim=1)(logit)
        logit2 = torch.nn.Softmax(dim=1)(logit2)
        if para.use_unequal == False or (para.use_unequal == True and use_fea_diff==False):
            unc_score = torch.abs(logit - logit2).sum(dim=1).unsqueeze(1)
        elif para.use_unequal == True and use_fea_diff==True:

            if para.unequal_type == 0:
                ini_unc_score3 = cal_unequal_score(logit, represent2, alpha, alpha2)
                ini_unc_score3_2 = cal_unequal_score(logit2, represent, alpha2, alpha)

                unc_score = 0.5 * torch.abs(ini_unc_score3 + ini_unc_score3_2).unsqueeze(1)
            elif para.unequal_type == 1 or para.unequal_type == 2:
                ini_unc_score3 = cal_unequal_score(logit, logit2, alpha, alpha2)

                unc_score = torch.abs(ini_unc_score3).unsqueeze(1)

            else:
                print('parameter is worng!')
                raise ValueError




        represent_set.append(represent.cpu().detach().numpy())

        logit_sfmx = logit  ####

        logit_sfset.append(logit_sfmx.cpu().detach().numpy())

        unc_score_set.append(unc_score.cpu().detach().numpy())



        y_pred = (torch.max(logit, 1)[1].view(query['text'].size()[0]).data).tolist()

        y_probs += [y_pred]



    logit_sfset = np.stack(np.array(logit_sfset), axis = 2)
    represent_set = np.stack(np.array(represent_set), axis = 2)
    logit_sfmean = logit_sfset.mean(axis=2)
    logit_sfvar = logit_sfset.var(axis=2).sum(axis=1)
    represent_setmean = represent_set.mean(axis=2)
    represent_setvar = represent_set.var(axis=2).sum(axis=1)
    # inst_weight = 1 - (torch.nn.Sigmoid()(logit_sfvar) - 0.5)*2
    if para.te_measure == '1max':
        drop_score = drop_entropy_emnlp(y_probs, mask_num, logit_sfmean, logit_sfvar, represent_setvar)
    elif para.te_measure == 'drop_entropy':
        drop_score = drop_entropy_naacl(y_probs, mask_num, logit_sfmean, logit_sfvar, represent_setvar)

    unc_score_set = np.stack(np.array(unc_score_set), axis = 2)
    unc_score_mean = unc_score_set.mean(axis=2)
    unc_score_mean = np.squeeze(unc_score_mean, axis=1)
    unc_score_mean = unc_score_mean.tolist()


    return drop_score, torch.from_numpy(logit_sfmean).float().cuda(), represent_setmean, unc_score_mean #(unc_score_mean is useless in the following calculation)




# few-shot evaluation
def few_test_one(task, model, args):
    '''
		Evaluate the model on a bag of sampled tasks. Return the mean accuracy
		and its std.
	'''

    print("using drop entropy ACL evaluation ...")

    assert args.individual_eval==True

    represent_all = torch.FloatTensor()
    target_all = torch.LongTensor()


    if True: # original is load the dataloader

        support, query = task


        # # Embedding the document
        XS1, TS_onehot1, alphaS1, XS2, TS_onehot2, alphaS2 = model['ebd'](support)
        YS = support['label']

        XQ1, TQ_onehot1, alphaQ1, XQ2, TQ_onehot2, alphaQ2 = model['ebd'](query)
        YQ = query['label']
        # Apply the classifier
        res = model['clf'](XS1, XS2, YS, XQ1, XQ2, YQ)

        C_YS, C_YQ, Q_pred1, Q_pred2, Q_logit1, Q_logit2 = res[0], res[1], res[2], res[3], res[4], res[5]

        model['ebd'].train()
        model['clf'].train()

        logit_sfset = []
        final_res_dict = {}
        final_res_dict['y_truth_scalar'] = []
        final_res_dict['y_pred_logit'] = []
        final_res_dict['y_uncertainty_ori'] = []
        final_res_dict['y_uncertainty_sec'] = []


        f1_score = {}
        f1_score['1_0'] = []

        f1_score['auroc'] = []
        f1_score['aupr'] = []


        for k in range(args.drop_num):

            y_pred = []
            y_truth = []
            uncertain_score_list = []

            sec_unc_mean_list = []

            if args.individual_eval:
                RES = uncertain_score(support, query, model, args.drop_num, args.drop_mask, args.use_sec_unc, para=args, use_fea_diff=False)
                score, logit_sf, represent, sec_unc = RES[0], RES[1], RES[2], RES[3]


                uncertain_score_list += score
                sec_unc_mean_list += sec_unc

                # save each softmax_res
                logit_sfset.append(logit_sf.cpu().detach().numpy())


            y_pred_cur = (torch.max(logit_sf, 1)[1].view(C_YQ.size()).data).tolist()    #只用了当前的数据，未用平均值
            y_truth_cur = C_YQ.data.tolist()

            y_pred += y_pred_cur
            y_truth += y_truth_cur


            # normalization
            uncertain_score_list = normalize_data(uncertain_score_list)

            sec_unc_mean_list = normalize_data(sec_unc_mean_list)


            logit_mean_all = logit_sf.cpu()



            if args.use_auc_roc:

                final_res_dict['y_truth_scalar'].append(y_truth)
                final_res_dict['y_pred_logit'].append(logit_mean_all.numpy())
                final_res_dict['y_uncertainty_ori'].append(uncertain_score_list)
                final_res_dict['y_uncertainty_sec'].append(sec_unc_mean_list)


            if args.use_idk:
                if args.individual_eval:
                    indices, L_sorted = zip(*sorted(enumerate(uncertain_score_list), key=itemgetter(1), reverse=False))
                    if args.use_auc_roc:
                        auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(
                            y_truth, logit_mean_all, uncertain_score_list)
                        f1_score['auroc'].append(auroc_score)
                        f1_score['aupr'].append(aupr_score)




                ### human idk
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
                    mid_f1_score = show_results(args.is_class_num_small, y_truth_cur, y_pred_cur, represent_all, target_all, writef_name=args.snapshot)
                    f1_score['1_0'].append(mid_f1_score)


            if not args.use_human_idk and not args.use_idk:

                f1_score = show_results(args.is_class_num_small, y_truth_cur, y_pred_cur, represent_all, target_all,
                                        writef_name=args.snapshot)





    vec_ratio = int(f1_score['1_0'].__len__() / f1_score['auroc'].__len__())

    new_1_0 = []
    for i in range(f1_score['auroc'].__len__()):
        mid_new_1_0= f1_score['1_0'][(vec_ratio * i) : (vec_ratio * i + vec_ratio)]
        new_1_0.append(mid_new_1_0)
    f1_score['1_0'] = np.array(new_1_0).mean(axis=0)
    f1_score['1_0_std'] = np.array(new_1_0).std(axis=0)




    f1_score['aupr_std'] = np.array(f1_score['aupr']).std()
    f1_score['auroc_std'] = np.array(f1_score['auroc']).std()
    f1_score['aupr'] = np.array(f1_score['aupr']).mean()
    f1_score['auroc'] = np.array(f1_score['auroc']).mean()


    return f1_score, final_res_dict


def few_test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()

    acc = {}
    acc['1_0'] = []
    acc['1_0_std'] = []

    if args.use_auc_roc:
        acc['auroc'] = []
        acc['aupr'] = []

        acc['auroc_std'] = []
        acc['aupr_std'] = []


    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    final_list_dict = []
    # jsq = 0
    for task in sampled_tasks:
        mid_res, mid_dict = few_test_one(task, model, args)
        final_list_dict.append(mid_dict)

        for key in mid_res.keys():
            acc[key].append(mid_res[key])



    # cal mean F1
    final_res = {}
    final_res['1_0'] = []
    final_res['1_0_std'] = []

    if args.use_auc_roc:
        final_res['auroc'] = []
        final_res['aupr'] = []


        final_res['auroc_std'] = []
        final_res['aupr_std'] = []


    for key in mid_res.keys():
        acc[key] = np.array(acc[key])
        final_res[key] = np.mean(acc[key], axis=0)

    print(final_res)
    try:
        f = open(args.snapshot + '_2.txt', 'a+')
    except:
        writef_name = './testres.txt'
        f = open(writef_name, 'a+')
    print('\n', sep='\t', file=f)

    for key in final_res.keys():
        print('below is the mean of '+ key, sep='\t', file=f)
        print(final_res[key], sep='\t', file=f)
    f.close()


    saved_dict_name = args.snapshot + '_2.npy'
    np.save(saved_dict_name, final_list_dict)
    new_final_list_dict = np.load(saved_dict_name, allow_pickle=True)
    return final_res
