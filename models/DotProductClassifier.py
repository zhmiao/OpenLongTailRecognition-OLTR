import torch.nn as nn
from utils import *

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, stage1_weights=False, dataset=None):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            self.fc_classifier_stage1 = init_weights(model=self.fc_classifier_stage1,
                                                     weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                                     classifier=True)
        else:
            print('Random initialized classifier weights.')
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None):
    print('Loading Dot Product Classifier.')
    return DotProduct_Classifier(num_classes, feat_dim, stage1_weights, dataset)