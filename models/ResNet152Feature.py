from models.ResNetFeature import *
from utils import *
        
def create_model(pretrain=True, weights='caffe', use_fc=False, dropout=None):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_selfatt=True, use_fc=use_fc, dropout=None)
    
    if pretrain:
        if weights == 'caffe':
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrained_weights/caffe_resnet152.pth',
                                     caffe=True)
        elif weights == 'sun_pre':
            print('Loading SUN Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrained_weights/sun_pretrained_resnet152.pth')

        elif weights == 'sun_pre_old':
            print('Loading SUN Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrained_weights/sun_pretrained_resnet152.pth.old')

        elif weights == 'places_pre':
            print('Loading PLACES Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./pretrained_weights/places_pretrained_resnet152.pth')
            # resnet152 = init_weights(model=resnet152,
            #                         weights_path='./pretrained_weights/final_model_checkpoint_step_places.pth.tar')
            #                          weights_path='./pretrained_weights/final_model_checkpoint_step_places.pth.tar')
    return resnet152
