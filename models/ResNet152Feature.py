from models.ResNetFeature import *
from utils import *
        
def create_model(pre_weights=None, use_selfatt=False, use_fc=False, dropout=None):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_selfatt=use_selfatt, use_fc=use_fc, dropout=None)
    
    if pre_weights == 'caffe':
        print('Loading Caffe Pretrained ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./pretrained_weights/caffe_resnet152.pth',
                                 caffe=True)

    elif pre_weights == 'sun':
        print('Loading SUN Stage 1 ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./pretrained_weights/sun_pretrained_resnet152.pth')

    elif pre_weights == 'places':
        print('Loading PLACES Stage 1 ResNet 152 Weights.')
        resnet152 = init_weights(model=resnet152,
                                 weights_path='./pretrained_weights/places_pretrained_resnet152.pth')

    else:
        print('No Pretrained Weights.')

    return resnet152
