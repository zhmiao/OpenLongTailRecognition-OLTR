from models.ResNetFeature import *
from utils import *
        
def create_model(pre_weights=None, use_selfatt=False, use_fc=False, dropout=None):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(Bottleneck, [1, 1, 1, 1], use_selfatt=use_selfatt, use_fc=use_fc, dropout=None)

    if pre_weights == 'imagenet_st1':
        print('Loading ImageNet Stage 1 ResNet 10 Weights.')
        resnet10 = init_weights(model=resnet10,
                                 weights_path='./logs/Imagenet_LT/stage1/final_model_checkpoint.pth')

    else:
        print('No Pretrained Weights For Feature Model.')

    return resnet10
