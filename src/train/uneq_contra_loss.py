import torch.nn as nn
import torch


class ContraLoss(nn.Module):
    def __init__(self):
        super(ContraLoss, self).__init__() #
        # self.noralize = nn.Softmax(dim=1)

    def normalize(self, x):
        return x / torch.norm(x, p='fro')

    def forward(self, p, z):
        z = z.detach()

        p = self.normalize(p)
        z = self.normalize(z)

        loss = -(p * z).sum(dim=1).mean()
        return loss

class UnEqualContraLoss(nn.Module):
    def __init__(self):
        super(UnEqualContraLoss, self).__init__() #
        self.noralize = nn.Softmax(dim=1)

    def calc_entropy(self, input_tensor):
        lsm = nn.LogSoftmax(dim=1)
        log_probs = lsm(input_tensor)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        entropy = -p_log_p.mean(dim=1)
        return entropy

    def forward(self, p, z, target_onehot_processed1, target_onehot_processed2, is_detach=True):

        if is_detach:
            z = z.detach()

        sign_index = (target_onehot_processed1 - target_onehot_processed2).sum(dim=1)



        p_Fnorm = self.calc_entropy(p)
        z_Fnorm = self.calc_entropy(z)

        ini_loss = -1 * (z_Fnorm - p_Fnorm)
        ini_loss2 = ini_loss * sign_index
        loss = torch.clamp(ini_loss2, min=0).sum()
        return loss