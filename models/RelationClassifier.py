import torch
import torch.nn as nn
from models.CosNormClassifier import CosNorm_Classifier

import pdb

class Relation_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000, stage1_weights=None):
        super(Relation_Classifier, self).__init__()
        self.num_classes = num_classes
        # self.fc_selector = nn.Linear(feat_dim, num_classes)
        self.fc_channel = nn.Linear(feat_dim, feat_dim)
        self.fc_classifier_stage1 = nn.Linear(feat_dim, num_classes)
        # self.fc_classifier_stage2 = nn.Linear(feat_dim, num_classes)
        
        self.stage1_weights = stage1_weights

        if self.stage1_weights:
            self.load_stage1_weights (self.stage1_weights)

        # self.fc_classifier_stage1.weight.requires_grad = False
        # self.fc_classifier_stage1.bias.requires_grad = False
        # self.fc_classifier_stage2.load_state_dict(weights)
        
        self.crossentropy_loss = nn.CrossEntropyLoss()
        # self.mse_loss = nn.MSELoss()
        # self.triplet_loss = TripletLoss(margin=0.5)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

    def load_weights(self, weights_dir):
        weights = torch.load(weights_dir)
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else self.fc_classifier_stage1.state_dict()[k] 
                   for k in self.fc_classifier_stage1.state_dict()}
        return weights

    def load_stage1_weights (self, stage1_weights):

        if stage1_weights == 'plain_fc':
            print('Loading ImageNet Pretrained Classifier Weights.')
            weights = self.load_weights('./pretrained_weights/imagenet_plain_resnet10_fc.pth')

        elif stage1_weights == 'sun_pre':
            print('Loading SUN Pretrained Classifier Weights.')
            weights = self.load_weights('./pretrained_weights/sun_pretrained_resnet152.pth')
            
        elif stage1_weights == 'sun_pre_old':
            print('Loading SUN Pretrained Classifier Weights.')
            weights = self.load_weights('./pretrained_weights/sun_pretrained_resnet152.pth.old')
                    
        elif stage1_weights == 'places_pre':
            print('Loading PLACES Pretrained Classifier Weights.')
            weights = self.load_weights('./pretrained_weights/places_pretrained_resnet152.pth')
            
        self.fc_classifier_stage1.load_state_dict(weights)
        
    def forward(self, x, labels, centers, class_count, *args):
        
        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # calculate current center
        x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
        centers_expand = centers.unsqueeze(0).expand(batch_size, -1, -1)
        dist_cur = torch.norm(x_expand - centers_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        confidence = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)
        # centers_cur = centers[labels_nn[:, 1], :]

        logits_stage1 = self.fc_classifier_stage1(x)
        prob_stage1 = logits_stage1.softmax(dim=1)
        centers_cur = torch.matmul(prob_stage1, centers)

        slow_feature = x

        # labels_onehot = torch.FloatTensor(batch_size, self.num_classes).cuda()
        # labels_onehot.zero_()
        # labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # self.loss_relation = self.num_classes * self.mse_loss(centers_selector, labels_onehot)
        # self.loss_relation = 0.5 * self.crossentropy_loss(logits_stage1, labels)

        # calculate channel selector
        channel_selector = self.fc_channel(x)
        channel_selector = channel_selector.tanh() 
        x = confidence * (x + channel_selector * centers_cur)

        fast_feature = channel_selector * centers_cur
        
        logits = self.cosnorm_classifier(x, labels)
        
        # class_count = torch.LongTensor(class_count)
        # class_type = torch.ones_like(class_count)
        # class_type[class_count > 100] = 0
        # class_type[class_count < 20] = 2
        # labels_type = class_type[labels]
        # if len(channel_selector) > 0 and len(torch.unique(labels)) > 2:
        #     self.loss_relation = self.loss_relation + 0.2 * self.triplet_loss(channel_selector, labels)

        self.loss_relation = 0.0

        return logits, self.loss_relation, [slow_feature, fast_feature]
    
def create_model(feat_dim=2048, num_classes=1000, stage1_weights=None):
    print('Loading Relation Classifier.')
    return Relation_Classifier(feat_dim, num_classes, stage1_weights)
