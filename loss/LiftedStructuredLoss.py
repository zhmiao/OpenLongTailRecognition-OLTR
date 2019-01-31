import torch
import torch.nn as nn
import pdb
"""
  "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
"""

class LiftedStructuredLoss (nn.Module):
    
    def __init__ (self, margin=5.0):
        super(LiftedStructuredLoss, self).__init__()
        self.margin = margin
        
    def forward(self, input, labels):

        loss = 0
        counter = 0
        
        bsz = input.shape[0]
        mag = (input ** 2).sum(1).expand(bsz, bsz)
        sim = input.mm(input.transpose(0, 1))

        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()

        for i in range(bsz):
            t_i = labels[i].item()

            for j in range(i + 1, bsz):
                t_j = labels[j].item()

                if t_i == t_j:
                    # Negative component
                    l_ni = (self.margin - dist[i][labels != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][labels != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()

                    # Positive component
                    l_p  = dist[i,j]

                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1

        return loss / (2 * counter)

def create_loss (margin=5.0):
    print('Loading lifted structured loss.')
    return LiftedStructuredLoss(margin=margin)
