"""
Geodesic Loss

Author: Lijie Yang
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np

from .builder import LOSSES

@LOSSES.register_module()
class GeodesicLoss(nn.Module):
    def __init__(self, reduction='mean'):
    #def __init__(self, reduction='mean'):
        super(GeodesicLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        m1 = m1.reshape(-1, 3, 3)# [batch,3,3]
        m2 = m2.reshape(-1, 3, 3)# [batch,3,3]
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # [batch,3,3]

        #m_trace = torch.zeros([batch]).to("cuda")
        #for i in range(batch):
        #    m_trace[i] = torch.trace(m[i,:,:])
        #cos = (m_trace - 1) / 2

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch))-0.001)
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1+0.001)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        # print(theta[:22])
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            # breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))
        else:
            return theta
