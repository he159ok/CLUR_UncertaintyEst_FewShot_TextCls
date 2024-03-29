import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embedding.wordebd import WORDEBD
from embedding.auxiliary.factory import get_embedding

import copy
import numpy as np

# from ..classifier.r2d2 import R2D2

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
        self.args.RatioForAlpha = 1/(1 - self.args.lowb)

        self.ebd = ebd
        self.aux = get_embedding(args)

        self.ebd_dim = self.ebd.embedding_dim

        input_dim = int(args.meta_idf) + self.aux.embedding_dim + \
            int(args.meta_w_target) + int(args.meta_iwf)

        if args.meta_ebd:
            # abalation use distributional signatures with word ebd may fail
            input_dim += self.ebd_dim

        if args.embedding == 'meta':
            self.rnn = RNN(input_dim, 25, 1, True, 0)

            self.seq = nn.Sequential(
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1),
                    )
        else:
            # use a mlp to predict the weight individually
            self.seq = nn.Sequential(
                nn.Linear(input_dim, 50),
                nn.ReLU(),
                nn.Dropout(self.args.dropout),
                nn.Linear(50, 1))



        if self.args.selfensemble:
            self.ebd2 = ebd2
            self.aux2 = get_embedding(args)


            if args.embedding == 'meta':
                self.rnn2 = RNN(input_dim, 25, 1, True, 0)

                self.seq2 = nn.Sequential(
                        nn.Dropout(self.args.dropout),
                        nn.Linear(50, 1),
                        )
            else:
                self.seq2 = nn.Sequential(
                    nn.Linear(input_dim, 50),
                    nn.ReLU(),
                    nn.Dropout(self.args.dropout),
                    nn.Linear(50, 1))


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
            @param data dictionary
                @key text: batch_size * max_text_len
                @key text_len: batch_size
            @param return_score bool
                set to true for visualization purpose

            @return output: batch_size * embedding_dim
        '''
        y1 = y2 = None

        bacthSize = data['text'].shape[0]
        alpha1 = alpha2 = None
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        y0, y2 = self.reidx_y(data1['label'], data2['label'])

        y0 = torch.zeros(bacthSize, self.args.way).scatter_(1, y0.unsqueeze(1).cpu(), 1).cuda()
        y2 = torch.zeros(bacthSize, self.args.way).scatter_(1, y2.unsqueeze(1).cpu(), 1).cuda()

        ebd = self.ebd(data1)
        scale = self.compute_score(data1, ebd)   #Eq 3
        ebd = torch.sum(ebd * scale, dim=1)

        if self.args.mixup and self.args.feature_aug_mode == None and (use_fea_diff or not self.args.mode =='test'):
            # (N, W, D)
            chooseInstID = torch.from_numpy(np.random.randint(0, bacthSize, size=(1, bacthSize))).t()
            chooseInstID_onehot = torch.zeros(bacthSize, bacthSize).scatter_(1, chooseInstID, 1).cuda()
            alpha = torch.from_numpy(np.random.rand(bacthSize) /self.args.RatioForAlpha).unsqueeze(1).float().cuda()

            # alpha = 0
            ebd = (1-alpha) * ebd + alpha * torch.mm(chooseInstID_onehot, ebd)

            if use_fea_diff == False:
                y1 = (1 - alpha) * y0.detach() + alpha * torch.mm(chooseInstID_onehot, y0.detach())

            # alpha1 = alpha # emnlp doesnot need to return the alpha value



        # strart process 2
        if self.args.mixup and self.args.selfensemble and self.args.feature_aug_mode == None and (use_fea_diff or not self.args.mode=='test'):

            ebd2 = self.ebd2(data2)
            scale2 = self.compute_score(data2, ebd2)  # Eq 3
            ebd2 = torch.sum(ebd2 * scale2, dim=1)

            ebd2 = (1 - alpha) * ebd2 + alpha * torch.mm(chooseInstID_onehot, ebd2)

            if use_fea_diff == False:
                y2 = (1 - alpha) * y0.detach() + alpha * torch.mm(chooseInstID_onehot, y0.detach())




        if return_score:
            return ebd, scale, y1, ebd2, scale2, y2

        if self.args.mode == "test":
            y1 = y2 = None

        if self.args.mode == "test" or not self.args.selfensemble:
            ebd2 = y2 = None

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
