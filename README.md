# Large-Scale Long-Tailed Recognition in an Open World

[[Project]](https://liuziwei7.github.io/projects/LongTail.html) [[Paper]](https://arxiv.org/abs/1904.05160)   

## Overview
`Open Long-Tailed Recognition (OLTR)` is the author's re-implementation of the long-tail recognizer described in:  
"[Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160)"   
[Ziwei Liu](https://liuziwei7.github.io/)<sup>\*</sup>,&nbsp; [Zhongqi Miao](https://github.com/zhmiao)<sup>\*</sup>,&nbsp; [Xiaohang Zhan](https://xiaohangzhan.github.io/),&nbsp; [Jiayun Wang](http://pwang.pw/),&nbsp; [Boqing Gong](http://boqinggong.info/),&nbsp; [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/)&nbsp; (CUHK & UC Berkeley / ICSI)&nbsp; 
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2019, **Oral Presentation**

<img src='./assets/intro.png' width=800>

Further information please contact [Zhongqi Miao](mailto:zhongqi.miao@berkeley.edu) and [Ziwei Liu](https://liuziwei7.github.io/).

## Update notifications
* 08/05/2019: Fixed a bug in `utils.py`. Update re-implemented ImageNet-LT weights at the end of this page.
* 05/02/2019: Fixed a bug in `run_network.py` so the models train properly. Update configuration file for Imagenet-LT stage 1 training so that the results from the paper can be reproduced.  

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [scikit-learn](https://scikit-learn.org/stable/)

## Data Preparation

<img src='./assets/dataset.png' width=800>

NOTE: Places-LT dataset have been updated since the first version. Please download again if you have the first version. 

- First, please download the [ImageNet_2014](http://image-net.org/index) and [Places_365](http://places2.csail.mit.edu/download.html) (256x256 version).
Please also change the `data_root` in `main.py` accordingly.

- Next, please download ImageNet-LT and Places-LT from [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf). Please put the downloaded files into the `data` directory like this:
```
data
  |--ImageNet_LT
    |--ImageNet_LT_open
    |--ImageNet_LT_train.txt
    |--ImageNet_LT_test.txt
    |--ImageNet_LT_val.txt
    |--ImageNet_LT_open.txt
  |--Places_LT
    |--Places_LT_open
    |--Places_LT_train.txt
    |--Places_LT_test.txt
    |--Places_LT_val.txt
    |--Places_LT_open.txt
```

## Download Caffe Pre-trained Models for Places_LT Stage_1 Training
* Caffe pretrained ResNet152 weights can be downloaded from [here](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtckNGQ2FLd25fa3c), and save the file to `./logs/caffe_resnet152.pth`

## Getting Started (Training & Testing)

<img src='./assets/pipeline.png' width=800>

### ImageNet-LT
- Stage 1 training:
```
python main.py --config ./config/ImageNet_LT/stage_1.py
```
- Stage 2 training:
```
python main.py --config ./config/ImageNet_LT/stage_2_meta_embedding.py
```
- Close-set testing:
```
python main.py --config ./config/ImageNet_LT/stage_2_meta_embedding.py --test
```
- Open-set testing (thresholding)
```
python main.py --config ./config/ImageNet_LT/stage_2_meta_embedding.py --test_open
```
- Test on stage 1 model
```
python main.py --config ./config/ImageNet_LT/stage_1.py --test
```

### Places-LT
- Stage 1 training:
```
python main.py --config ./config/Places_LT/stage_1.py
```
- Stage 2 training:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding.py
```
- Close-set testing:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding.py --test
```
- Open-set testing (thresholding)
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding.py --test_open
```

## Benchmarks and Model Zoo

### ImageNet-LT Open-Set Setting

|   Backbone  |    Many-Shot   |  Medium-Shot  |   Few-Shot  |  F-Measure  |      Download      |
| :---------: | :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-10  |      41.2      |      33.6     |    17.5     |     47.4    |     [model](https://drive.google.com/file/d/1w4oZ9Jmaca-NnO_tSvPaPue3u9jh2em-/view?usp=sharing)      |

### Places-LT Open-Set Setting

|   Backbone  |    Many-Shot   |  Medium-Shot  |   Few-Shot  |  F-Measure  |      Download      |
| :---------: | :------------: | :-----------: | :---------: | :---------: | :----------------: |
| ResNet-152  |      38.5      |     37.0      |    21.8     |     48.5    |     [model](https://drive.google.com/file/d/1dNvceMdxEgHWopU2lJEhH_qdBTGIWySY/view?usp=sharing)      |

## CAUTION
The current code was prepared using single GPU. The use of multi-GPU can cause problems. 

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.
```
@inproceedings{openlongtailrecognition,
  title={Large-Scale Long-Tailed Recognition in an Open World},
  author={Liu, Ziwei and Miao, Zhongqi and Zhan, Xiaohang and Wang, Jiayun and Gong, Boqing and Yu, Stella X.},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
