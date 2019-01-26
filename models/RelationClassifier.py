import torch
import torch.nn as nn
from models.CosNormClassifier import CosNorm_Classifier
from utils import *

import pdb

class Relation_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000, stage1_weights=None):
        super(Relation_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_channel = nn.Linear(feat_dim, feat_dim)
        self.fc_classifier_stage1 = nn.Linear(feat_dim, num_classes)

        # if self.stage1_weights:
        #     self.load_stage1_weights (self.stage1_weights)

        if stage1_weights == 'imagenet':
            print('Loading ImageNet Stage 1 Classifier Weights.')
            self.fc_classifier_stage1 = init_weights(model=self.fc_classifier_stage1,
                                                     weights_path='./pretrained_weights/imagenet_plain_resnet10_fc.pth',
                                                     classifier=True)
        
        elif stage1_weights == 'sun':
            print('Loading SUN Stage 1 Classifier Weights.')
            self.fc_classifier_stage1 = init_weights(model=self.fc_classifier_stage1,
                                                     weights_path='./pretrained_weights/sun_pretrained_resnet152.pth',
                                                     classifier=True)

        elif stage1_weights == 'places':
            print('Loading PLACES Stage 1 Classifier Weights.')
            self.fc_classifier_stage1 = init_weights(model=self.fc_classifier_stage1,
                                                     weights_path='./pretrained_weights/places_pretrained_resnet152.pth',
                                                     classifier=True)
        
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

    # def load_weights(self, weights_dir):
    #     weights = torch.load(weights_dir)
    #     weights = weights['state_dict_best']['classifier']
    #     weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else self.fc_classifier_stage1.state_dict()[k] 
    #                for k in self.fc_classifier_stage1.state_dict()}
    #     return weights

    # def load_stage1_weights (self, stage1_weights):

    #     if stage1_weights == 'plain_fc':
    #         print('Loading ImageNet Pretrained Classifier Weights.')
    #         weights = self.load_weights('./pretrained_weights/imagenet_plain_resnet10_fc.pth')

    #     elif stage1_weights == 'sun_pre':
    #         print('Loading SUN Pretrained Classifier Weights.')
    #         weights = self.load_weights('./pretrained_weights/sun_pretrained_resnet152.pth')
            
    #     elif stage1_weights == 'sun_pre_old':
    #         print('Loading SUN Pretrained Classifier Weights.')
    #         weights = self.load_weights('./pretrained_weights/sun_pretrained_resnet152.pth.old')
                    
    #     elif stage1_weights == 'places_pre':
    #         print('Loading PLACES Pretrained Classifier Weights.')
    #         weights = self.load_weights('./pretrained_weights/places_pretrained_resnet152.pth')
            
    #     self.fc_classifier_stage1.load_state_dict(weights)
        
    def forward(self, x, labels, centers, class_count, *args):
        
        slow_feature = x

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # calculate current center
        x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
        centers_expand = centers.unsqueeze(0).expand(batch_size, -1, -1)
        dist_cur = torch.norm(x_expand - centers_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        confidence = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        logits_stage1 = self.fc_classifier_stage1(x)
        prob_stage1 = logits_stage1.softmax(dim=1)
        centers_cur = torch.matmul(prob_stage1, centers)

        # calculate channel selector
        channel_selector = self.fc_channel(x)
        channel_selector = channel_selector.tanh() 
        x = confidence * (x + channel_selector * centers_cur)

        fast_feature = channel_selector * centers_cur
        
        logits = self.cosnorm_classifier(x, labels)

        return logits, [slow_feature, fast_feature]
    
def create_model(feat_dim=2048, num_classes=1000, stage1_weights=None):
    print('Loading Relation Classifier.')
    return Relation_Classifier(feat_dim, num_classes, stage1_weights)
