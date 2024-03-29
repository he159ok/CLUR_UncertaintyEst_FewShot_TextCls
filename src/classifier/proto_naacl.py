import torch
import torch.nn as nn
from classifier.base import BASE


class PROTO(BASE):
    '''
        PROTOTIPICAL NETWORK FOR FEW SHOT LEARNING
    '''
    def __init__(self, ebd_dim, args):
        super(PROTO, self).__init__(args)
        self.ebd_dim = ebd_dim
        self.args = args

        if args.embedding == 'meta':
            self.mlp = None
            print('No MLP')
        else:
            self.mlp = self._init_mlp(
                    self.ebd_dim, self.args.proto_hidden, self.args.dropout)

        # contrastive learning
        self.class_num = args.way
        self.project_dim = 100

        self.fc1 = nn.Sequential(
            nn.Linear(self.class_num, self.project_dim),
            nn.BatchNorm1d(self.project_dim),
            # nn.ReLU(),
            nn.Dropout(self.args.dropout_MC),
            nn.Linear(self.project_dim, self.class_num)
            # nn.Softmax(dim=1)
        )  # SimSiam中的predictor

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self,XS, XS2, YS, XQ, XQ2, YQ, mixup_y = None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        if self.mlp is not None:
            XS = self.mlp(XS)
            XQ = self.mlp(XQ)

        YS, YQ = self.reidx_y(YS, YQ)

        prototype = self._compute_prototype(XS, YS)

        pred1 = -self._compute_l2(prototype, XQ)

        C_YS = YS
        C_YQ = YQ

        if self.args.use_no_further_projection == True and self.args.metric == True:
            raise ValueError
        if self.args.use_no_further_projection == True and self.args.metric == False:
            return C_YS, C_YQ, XQ, pred1
        else:
            logit1 = self.fc1(pred1)

            return C_YS, C_YQ, pred1, logit1

