from models.ResNetFeature import *
from utils import *
        
def create_model(pre_weights=None, use_selfatt=False, use_fc=False, dropout=None):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_selfatt=use_selfatt, use_fc=use_fc, dropout=None)
    
    if pre_weights == 'caffe':
        print('Loading Caffe Pretrained ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./logs/caffe_resnet152.pth',
                                 caffe=True)

    elif pre_weights == 'sun_st1':
        print('Loading SUN Stage 1 ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./logs/SUN_LT/stage1/final_model_checkpoint.pth')

    elif pre_weights == 'places_st1':
        print('Loading PLACES Stage 1 ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./logs/Places_LT/stage1/final_model_checkpoint.pth')

    else:
        print('No Pretrained Weights For Feature Model.')

    return resnet152
