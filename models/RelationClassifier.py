import torch
import torch.nn as nn
from models.CosNormClassifier import CosNorm_Classifier
from utils import *

import pdb

class Relation_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(Relation_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_channel = nn.Linear(feat_dim, feat_dim)
        self.fc_classifier_stage1 = nn.Linear(feat_dim, num_classes)
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)
        
    def forward(self, x, centers, *args):
        
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
        
        logits = self.cosnorm_classifier(x)

        return logits, [slow_feature, fast_feature]
    
def create_model(feat_dim=2048, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Relation Classifier.')
    clf = Relation_Classifier(feat_dim, num_classes)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc_classifier_stage1 = init_weights(model=clf.fc_classifier_stage1,
                                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                                    classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf