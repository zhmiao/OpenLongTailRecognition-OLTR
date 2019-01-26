import torch.nn as nn
from utils import *

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, stage1_weights=None):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

        # weights = torch.load('./pretrained_weights/final_model_checkpoint_step.pth.tar')
        # weights = weights['state_dict_best']['classifier']
        # weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else self.fc.state_dict()[k] 
        #            for k in self.fc.state_dict()}

        # self.fc.load_state_dict(weights)

        if stage1_weights == 'imagenet':
            print('Loading ImageNet Stage 1 Classifier Weights.')
            self.fc = init_weights(model=self.fc,
                                   weights_path='./pretrained_weights/imagenet_plain_resnet10_fc.pth',
                                   classifier=True)
            
        elif stage1_weights == 'sun':
            print('Loading SUN Stage 1 Classifier Weights.')
            self.fc = init_weights(model=self.fc,
                                   weights_path='./pretrained_weights/sun_pretrained_resnet152.pth',
                                   classifier=True)

        elif stage1_weights == 'places':
            print('Loading PLACES Stage 1 Classifier Weights.')
            self.fc = init_weights(model=self.fc,
                                   weights_path='./pretrained_weights/places_pretrained_resnet152.pth',
                                   classifier=True)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None
    
def create_model(feat_dim, num_classes=1000, pretrain=False):
    print('Loading Dot Product Classifier.')
    return DotProduct_Classifier(num_classes, feat_dim)