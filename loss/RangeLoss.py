import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd.function import Function

import pdb

class RangeLoss(nn.Module):
    """Range loss.
    
    Reference:
    Zhang et al. Range Loss for Deep Face Recognition with Long-tail. CVPR 2017.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=1000, feat_dim=512, margin=5.0, alpha=1.0, beta=1.0):
        super(RangeLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.centers = Parameter(torch.Tensor(num_classes, feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        
        mag = (x ** 2).sum(1).expand(batch_size, batch_size)
        sim = x.mm(x.transpose(0, 1))

        distmat_intra = (mag + mag.transpose(0, 1) - 2 * sim)
        distmat_intra = torch.nn.functional.relu(distmat_intra).sqrt()

        labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
        mask = labels_expand.eq(labels_expand.t())

        dist_intra = []
        for i in range(batch_size):
            value = distmat_intra[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist_intra.append(value)

        dist_intra = torch.cat(dist_intra)
        distmat_intra_descend, _ = torch.sort(dist_intra, descending=True) 

        distmat_inter = 2 * torch.pow(self.centers, 2).sum(dim=1, keepdim=False)
        distmat_inter = distmat_inter - 2 * torch.matmul(self.centers, self.centers.t())
        distmat_inter = distmat_inter.view(1, -1).squeeze(0)
        distmat_inter = distmat_inter.clamp(min=1e-12, max=1e+12) # for numerical stability
        distmat_inter_ascend, _ = torch.sort(distmat_inter, descending=False)

        k = 2
        loss_intra = k / (1 / (distmat_intra_descend[0]) + 1 / (distmat_intra_descend[1]))
        loss_inter = torch.clamp(self.margin - distmat_inter_ascend[0], 0.0, 1e6)

        loss = self.alpha * loss_intra + self.beta * loss_inter

        return loss

    
def create_loss (feat_dim=512, num_classes=1000):
    print('Loading Range Loss.')
    return RangeLoss(num_classes, feat_dim)