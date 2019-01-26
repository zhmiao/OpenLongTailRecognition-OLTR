from models.ResNetFeature import *
from utils import *
        
def create_model(pre_weights=None, use_selfatt=False, use_fc=False, dropout=None):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(Bottleneck, [1, 1, 1, 1], use_selfatt=use_selfatt, use_fc=use_fc, dropout=None)

    if pre_weights == 'imagenet':
        print('Loading ImageNet Stage 1 ResNet 10 Weights.')
        resnet10 = init_weights(model=resnet10,
                                 weights_path='./pretrained_weights/sun_pretrained_resnet10.pth')

    else:
        print('No Pretrained Weights.')

    return resnet10
