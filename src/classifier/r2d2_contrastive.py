import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE




class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim
        self.args = args
        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))

        self.lam2 = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta2 = nn.Parameter(torch.tensor(1, dtype=torch.float))

        # contrastive learning
        self.class_num = args.way
        self.project_dim = 100
        if self.args.contrastive:
            self.fc1 = nn.Sequential(
                nn.Linear(self.class_num, self.project_dim),
                nn.BatchNorm1d(self.project_dim),
                nn.Dropout(self.args.dropout_MC),
                nn.Linear(self.project_dim, self.class_num)
            )

            self.fc1_2 = nn.Sequential(
                nn.Linear(self.class_num, self.project_dim),
                nn.BatchNorm1d(self.project_dim),
                nn.Dropout(self.args.dropout_MC),
                nn.Linear(self.project_dim, self.class_num)
            )

        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _compute_w2(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam2) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, XS2, YS, XQ, XQ2, YQ, mixup_y = None, mixup_y2 = None):

        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        if self.args.mode == "test":
            YS_onehot = self._label2onehot(YS)

            W = self._compute_w(XS, YS_onehot)
            W2 = self._compute_w2(XS2, YS_onehot)
        else:
            if self.args.use_original_ref:
                YS_onehot = self._label2onehot(YS)
                W = self._compute_w(XS, YS_onehot)
                W2 = self._compute_w2(XS2, YS_onehot)
            else:
                W = self._compute_w(XS, mixup_y)
                W2 = self._compute_w2(XS2, mixup_y2)

        pred1 = (10.0 ** self.alpha) * XQ @ W + self.beta

        pred2 = (10.0 ** self.alpha2) * XQ2 @ W2 + self.beta2


        C_YS = YS
        C_YQ = YQ

        if self.args.use_no_further_projection:
            assert (self.args.unequal_type != 0)
            return C_YS, C_YQ, XQ, XQ2, pred1, pred2
        else:
            logit1 = self.fc1(pred1)
            logit2 = self.fc1_2(pred2)



        return C_YS, C_YQ, pred1, pred2, logit1, logit2
        # Calibrated YS



    def visualize(self, XS, YS, XQ, YQ, mixup_y = None, mixup_y2 = None):

        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        if self.args.mode == "test":
            YS_onehot = self._label2onehot(YS)

            W = self._compute_w(XS, YS_onehot)

        else:
            if self.args.use_original_ref:
                YS_onehot = self._label2onehot(YS)
                W = self._compute_w(XS, YS_onehot)
            else:
                W = self._compute_w(XS, mixup_y)

        pred1 = (10.0 ** self.alpha) * XQ @ W + self.beta


        C_YS = YS
        C_YQ = YQ

        if self.args.use_no_further_projection:
            assert (self.args.unequal_type != 0)
            return C_YS, C_YQ, XQ, pred1
        else:
            logit1 = self.fc1(pred1)



        return C_YS, C_YQ, pred1, logit1
        # Calibrated YS
