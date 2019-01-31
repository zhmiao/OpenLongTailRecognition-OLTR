# Large-scale long-tailed recognition in an open world

Caffe pretrained ResNet152 weights can be downloaded from [here](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtckNGQ2FLd25fa3c), and save the file to `.logs/caffe_resnet152.pth`

## Places (*Now only for testing*)
- Stage 1 training:
```
python main.py --config ./config/Places_LT/stage_1_test.py
```
- Stage 2 training:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py --use_step True
```
- Close-set testing:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py --use_step True --test True
```
- Open-set testing (thresholding)
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py --use_step True --test True --test_open True
```
