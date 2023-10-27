import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embedding.wordebd import WORDEBD
from embedding.auxiliary.factory import get_embedding

import copy
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional,
            dropout):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                bidirectional=bidirectional, dropout=dropout)


    def _sort_tensor(self, input, lengths):
        '''
        pack_padded_sequence  requires the length of seq be in descending order
        to work.
        Returns the sorted tensor, the sorted seq length, and the
        indices for inverting the order.

        Input:
                input: batch_size, seq_len, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, seq_len, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        return sorted_input, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        '''
        Recover the origin order

        Input:
                input:        batch_size-num_zero, seq_len, hidden_dim
                invert_order: batch_size
                num_zero
        Output:
                out:   batch_size, seq_len, *
        '''
        if num_zero == 0:
            input = input[invert_order]

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros((num_zero, dim1, dim2), device=input.device,
                    dtype=input.dtype)
            input = torch.cat((input, zero), dim=0)
            input = input[invert_order]

        return input

    def forward(self, text, text_len):
        '''
        Input: text, text_len
            text       Variable  batch_size * max_text_len * input_dim
            text_len   Tensor    batch_size

        Output: text
            text       Variable  batch_size * max_text_len * output_dim
        '''
        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_text, sort_len, invert_order, num_zero = self._sort_tensor(input=text, lengths=text_len)
        text = pack_padded_sequence(sort_text, lengths=sort_len.cpu().numpy(), batch_first=True)


        # Run through the word level RNN
        text, _ = self.rnn(text)         # batch_size, max_doc_len, args.word_hidden_size

        # Unpack the output, and invert the sorting
        text = pad_packed_sequence(text, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        text = self._unsort_tensor(text, invert_order, num_zero) # batch_size, max_doc_len, rnn_size

        return text


class META(nn.Module):
    def __init__(self, ebd, ebd2, args):
        super(META, self).__init__()

        self.args = args
        if self.args.lowb >= 1:
            self.args.lowb = 0.999999
        self.args.RatioForAlpha = 1/(1 - self.args.lowb)  #$\Phi_1$

        if self.args.larger_augmargin:
            self.larger_w = self.args.lowb - self.args.augmargin -self.args.sec_lowb
            self.larger_b = 1 - self.args.lowb + self.args.augmargin



        self.ebd = ebd
        self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        input_dim = int(args.meta_idf) + self.aux.embedding_dim + \
            int(args.meta_w_target) + int(args.meta_iwf)

        if args.meta_ebd:
            input_dim += self.ebd_dim

        if args.embedding == 'meta':
            self.rnn = RNN(input_dim, 25, 1, True, 0)

            self.seq = nn.Sequential(
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1),
                    )
        else:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(50, 1))


        self.ebd2 = ebd2
        self.aux2 = get_embedding(args)



        if args.embedding == 'meta':
            self.rnn2 = RNN(input_dim, 25, 1, True, 0)

            self.seq2 = nn.Sequential(
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1),
                    )
        else:
            # use a mlp to predict the weight individually
            self.seq2 = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(50, 1))

        self.dropout1 = nn.Dropout(self.args.dropout)
        self.dropout2 = nn.Dropout(self.args.dropout)

    def reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def forward(self, data, return_score=False, use_fea_diff=False):
        '''

        '''
        y1 = y2 = None

        cur_batchsize = data['text'].shape[0]
        alpha1 = alpha2 = None
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        y0, y2 = self.reidx_y(data1['label'], data2['label'])

        y0 = torch.zeros(cur_batchsize, self.args.way).scatter_(1, y0.unsqueeze(1).cpu(), 1).cuda()
        y2 = torch.zeros(cur_batchsize, self.args.way).scatter_(1, y2.unsqueeze(1).cpu(), 1).cuda()

        if self.args.contrastive and self.args.feature_aug_mode != None and (use_fea_diff or not self.args.mode =='test'):

            first_smaller_alpha = None
            if self.args.larger_augmargin:
                cur_flip = np.random.rand(1)[0]
                if cur_flip < 0.5:
                    first_smaller_alpha = True
                else:
                    first_smaller_alpha = False

            if self.args.AugLevel == 'sam':
                if first_smaller_alpha == True or self.args.larger_augmargin == False:
                    alpha = torch.from_numpy(np.random.rand(cur_batchsize) / self.args.RatioForAlpha).unsqueeze(
                    1).float().cuda()
                else:
                    alpha = torch.from_numpy(np.random.rand(cur_batchsize) * self.larger_w + self.larger_b).unsqueeze(
                        1).float().cuda()
            elif self.args.AugLevel == 'bat':
                if first_smaller_alpha == True or self.args.larger_augmargin == False:
                    ini_randval = np.random.rand(1) / self.args.RatioForAlpha
                else:
                    ini_randval = np.random.rand(1) * self.larger_w + self.larger_b
                mid_alpha = ini_randval.repeat(cur_batchsize, axis=None)
                alpha = torch.from_numpy(mid_alpha).unsqueeze(1).float().cuda()
            h_num, w_num = data1['text'].shape
            if self.args.feature_aug_mode == 'cutoff':

                filter_mat1 = torch.ones_like(data1['text']).cuda()

                if self.args.AugLevel == 'sam':
                    for i in range(h_num):
                        mid_order = np.arange(w_num)
                        mid_pick = np.array(list(range(int(w_num * alpha[i]))))
                        reshuffle_len = len(mid_pick)
                        alpha[i] = reshuffle_len * 1.0 / w_num
                        np.random.shuffle(mid_pick)
                        np.random.shuffle(mid_order)
                        filter_mat1[i, mid_order[0:reshuffle_len]] = 0
                elif self.args.AugLevel == 'bat':
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * ini_randval[0]))))
                    reshuffle_len = len(mid_pick)
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)
                    filter_mat1[:, mid_order[0:reshuffle_len]] = 0

                data1['text'] = torch.mul(data1['text'], filter_mat1)

                if use_fea_diff == False:
                    y1 = y0.detach() * (1 - alpha)

            if self.args.feature_aug_mode == 'shuffle':
                for i in range(h_num):
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * alpha[i]))))
                    reshuffle_len = len(mid_pick)
                    alpha[i] = reshuffle_len * 1.0 / w_num
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)
                    shuffle_order1 = mid_order.copy()
                    shuffle_order2 = mid_order.copy()
                    shuffle_order2[0:reshuffle_len] = mid_order.copy()[0:reshuffle_len][::-1]
                    data1['text'][i, list(shuffle_order1)] = data1['text'][i, list(shuffle_order2)]

                if use_fea_diff == False:
                    y1 = y0.detach() * (1 - alpha)

            alpha1 = alpha


        if self.args.contrastive and self.args.feature_aug_mode != None and (use_fea_diff or not self.args.mode=='test'):
            if self.args.use_unequal:
                if self.args.AugLevel == 'sam':
                    if first_smaller_alpha == False or self.args.larger_augmargin == False:
                        alpha2 = torch.from_numpy(np.random.rand(cur_batchsize) / self.args.RatioForAlpha).unsqueeze(1).float().cuda()
                    else:
                        alpha2 = torch.from_numpy(np.random.rand(cur_batchsize) * self.larger_w + self.larger_b).unsqueeze(
                            1).float().cuda()
                elif self.args.AugLevel == 'bat':
                    if first_smaller_alpha == False or self.args.larger_augmargin == False:
                        ini_randval2 = np.random.rand(1) / self.args.RatioForAlpha
                    else:
                        ini_randval2 = np.random.rand(1) * self.larger_w + self.larger_b
                    mid_alpha2 = ini_randval2.repeat(cur_batchsize, axis=None)
                    alpha2 = torch.from_numpy(mid_alpha2).unsqueeze(1).float().cuda()
            elif self.args.use_equal:
                if self.args.AugLevel == 'sam':
                    alpha2 = alpha
                elif self.args.AugLevel == 'bat':
                    ini_randval2 = ini_randval
                    alpha2 = alpha

            h_num, w_num = data2['text'].shape
            if self.args.feature_aug_mode == 'cutoff':
                filter_mat2 = torch.ones_like(data2['text']).cuda()

                if self.args.AugLevel == 'sam':
                    for i in range(h_num):
                        mid_order = np.arange(w_num)
                        mid_pick = np.array(list(range(int(w_num * alpha2[i]))))
                        reshuffle_len = len(mid_pick)
                        if self.args.use_unequal:
                            alpha2[i] = reshuffle_len * 1.0 / w_num
                        elif self.args.use_equal:
                            pass
                        np.random.shuffle(mid_pick)
                        np.random.shuffle(mid_order)
                        filter_mat2[i, mid_order[0:reshuffle_len]] = 0
                elif self.args.AugLevel == 'bat':
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * ini_randval2[0]))))
                    reshuffle_len = len(mid_pick)
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)

                    filter_mat2[:, mid_order[0:reshuffle_len]] = 0

                data2['text'] = torch.mul(data2['text'], filter_mat2)
                if use_fea_diff == False:
                    y2 = y0.detach() * (1 - alpha2)

            elif self.args.feature_aug_mode == 'shuffle':


                alpha2 = torch.from_numpy(np.random.rand(cur_batchsize) / self.args.RatioForAlpha).unsqueeze(
                    1).float().cuda()

                for i in range(h_num):
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * alpha[i]))))
                    reshuffle_len = len(mid_pick)
                    if self.args.use_unequal:
                        alpha2[i] = reshuffle_len * 1.0 / w_num
                    elif self.args.use_equal:
                        pass
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)
                    shuffle_order1 = mid_order.copy()
                    shuffle_order2 = mid_order.copy()
                    shuffle_order2[0:reshuffle_len] = mid_order.copy()[0:reshuffle_len][::-1]
                    data2['text'][i, list(shuffle_order1)] = data2['text'][i, list(shuffle_order2)]


                if use_fea_diff == False:
                    y2 = y0.detach() * (1 - alpha2)



        ebd = self.ebd(data1)

        scale = self.compute_score(data1, ebd)

        ebd = torch.sum(ebd * scale, dim=1)

        ebd2 = self.ebd2(data2)

        scale2 = self.compute_score(data2, ebd2)

        ebd2 = torch.sum(ebd2 * scale2, dim=1)

        if self.args.embedding_dropout:
            ebd = self.dropout1(ebd)
            ebd2 = self.dropout2(ebd2)





        if return_score: ### it is always False in my setting
            return ebd, scale, y1, ebd2, scale2, y2

        if self.args.mode == "test":
            y1 = y2 = None

        return ebd, y1, alpha1, ebd2, y2, alpha2

    def _varlen_softmax(self, logit, text_len):
        '''
            Compute softmax for sentences with variable length
            @param: logit: batch_size * max_text_len
            @param: text_len: batch_size

            @return: score: batch_size * max_text_len
        '''
        logit = torch.exp(logit)
        mask = torch.arange(
                logit.size()[-1], device=logit.device,
                dtype=text_len.dtype).expand(*logit.size()
                        ) < text_len.unsqueeze(-1)

        logit = mask.float() * logit
        score = logit / torch.sum(logit, dim=1, keepdim=True)

        return score

    def compute_score(self, data, ebd, return_stats=False):
        '''
            Compute the weight for each word

            @param data dictionary
            @param return_stats bool
                return statistics (input and output) for visualization purpose
            @return scale: batch_size * max_text_len * 1
        '''

        # preparing the input for the meta model
        x = self.aux(data)
        if self.args.meta_idf:
            idf = F.embedding(data['text'], data['idf']).detach()
            x = torch.cat([x, idf], dim=-1)

        if self.args.meta_iwf:
            iwf = F.embedding(data['text'], data['iwf']).detach()
            x = torch.cat([x, iwf], dim=-1)

        if self.args.meta_ebd:
            x = torch.cat([x, ebd], dim=-1)

        if self.args.meta_w_target:
            if self.args.meta_target_entropy:
                w_target = ebd @ data['w_target']
                w_target = F.softmax(w_target, dim=2) * F.log_softmax(w_target,
                        dim=2)
                w_target = -torch.sum(w_target, dim=2, keepdim=True)
                w_target = 1.0 / w_target
                x = torch.cat([x, w_target.detach()], dim=-1)
            else:
                # for rr approxmiation, use the max weight to approximate
                # task-specific importance
                w_target = torch.abs(ebd @ data['w_target'])
                w_target = w_target.max(dim=2, keepdim=True)[0]
                x = torch.cat([x, w_target.detach()], dim=-1)

        if self.args.embedding == 'meta':
            # run the LSTM
            hidden = self.rnn(x, data['text_len'])
        else:
            hidden = x

        # predict the logit
        logit = self.seq(hidden).squeeze(-1)  # batch_size * max_text_len

        score = self._varlen_softmax(logit, data['text_len']).unsqueeze(-1)

        if return_stats:
            return score.squeeze(), idf.squeeze(), w_target.squeeze()
        else:
            return score

    def visualize(self, data, return_score=False, use_fea_diff=False, lowb=0.9):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
                @key text_len: batch_size
            @param return_score bool
                set to true for visualization purpose

            @return output: batch_size * embedding_dim
        '''
        y1 = y2 = None
        ratio_for_alpha = 1 / (1 - lowb)

        cur_batchsize = data['text'].shape[0]
        alpha1 = alpha2 = None
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        y0, y2 = self.reidx_y(data1['label'], data2['label'])

        y0 = torch.zeros(cur_batchsize, self.args.way).scatter_(1, y0.unsqueeze(1).cpu(), 1).cuda()
        y2 = torch.zeros(cur_batchsize, self.args.way).scatter_(1, y2.unsqueeze(1).cpu(), 1).cuda()

        if self.args.contrastive and self.args.feature_aug_mode != None and (
                use_fea_diff or not self.args.mode == 'test'):
            # (N, W, D)

            first_smaller_alpha = None
            if self.args.larger_augmargin:
                cur_flip = np.random.rand(1)[0]
                if cur_flip < 0.5:
                    first_smaller_alpha = True
                else:
                    first_smaller_alpha = False

            if self.args.AugLevel == 'sam':
                if first_smaller_alpha == True or self.args.larger_augmargin == False:
                    # alpha = torch.from_numpy(np.random.rand(cur_batchsize) / ratio_for_alpha).unsqueeze(
                    #     1).float().cuda()
                    alpha = torch.from_numpy(np.ones(cur_batchsize) / ratio_for_alpha).unsqueeze(
                        1).float().cuda()
                else:
                    # alpha = torch.from_numpy(np.random.rand(cur_batchsize) * self.larger_w + self.larger_b).unsqueeze(
                    #     1).float().cuda()
                    alpha = torch.from_numpy(np.ones(cur_batchsize) / ratio_for_alpha).unsqueeze(
                        1).float().cuda()
            elif self.args.AugLevel == 'bat':
                if first_smaller_alpha == True or self.args.larger_augmargin == False:
                    # ini_randval = np.random.rand(1) / ratio_for_alpha
                    ini_randval = np.ones(1) / ratio_for_alpha
                else:
                    # ini_randval = np.random.rand(1) * self.larger_w + self.larger_b
                    ini_randval = np.ones(1) / ratio_for_alpha
                mid_alpha = ini_randval.repeat(cur_batchsize, axis=None)
                alpha = torch.from_numpy(mid_alpha).unsqueeze(1).float().cuda()
            # aug_target1 = y0.detach() * (1 - alpha)
            h_num, w_num = data1['text'].shape
            if self.args.feature_aug_mode == 'cutoff':

                filter_mat1 = torch.ones_like(data1['text']).cuda()

                if self.args.AugLevel == 'sam':
                    for i in range(h_num):
                        mid_order = np.arange(w_num)
                        mid_pick = np.array(list(range(int(w_num * alpha[i]))))
                        reshuffle_len = len(mid_pick)
                        alpha[i] = reshuffle_len * 1.0 / w_num
                        np.random.shuffle(mid_pick)
                        np.random.shuffle(mid_order)
                        filter_mat1[i, mid_order[0:reshuffle_len]] = 0
                elif self.args.AugLevel == 'bat':
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * ini_randval[0]))))
                    reshuffle_len = len(mid_pick)
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)
                    filter_mat1[:, mid_order[0:reshuffle_len]] = 0

                data1['text'] = torch.mul(data1['text'], filter_mat1)
                # x = self.embed(x0)

                if use_fea_diff == False:
                    y1 = y0.detach() * (1 - alpha)

            if self.args.feature_aug_mode == 'shuffle':
                for i in range(h_num):
                    mid_order = np.arange(w_num)
                    mid_pick = np.array(list(range(int(w_num * alpha[i]))))
                    reshuffle_len = len(mid_pick)
                    alpha[i] = reshuffle_len * 1.0 / w_num
                    np.random.shuffle(mid_pick)
                    np.random.shuffle(mid_order)
                    shuffle_order1 = mid_order.copy()
                    shuffle_order2 = mid_order.copy()
                    shuffle_order2[0:reshuffle_len] = mid_order.copy()[0:reshuffle_len][::-1]
                    data1['text'][i, list(shuffle_order1)] = data1['text'][i, list(shuffle_order2)]

                if use_fea_diff == False:
                    y1 = y0.detach() * (1 - alpha)

            alpha1 = alpha


        ebd = self.ebd(data1)

        scale = self.compute_score(data1, ebd)  # Eq 3

        ebd = torch.sum(ebd * scale, dim=1)


        if self.args.embedding_dropout:
            ebd = self.dropout1(ebd)

        if return_score:
            return ebd, scale, y1

        if self.args.mode == "test":
            y1 = y2 = None

        return ebd, y1, alpha1