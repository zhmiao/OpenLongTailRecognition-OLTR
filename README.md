# Large-Scale Long-Tailed Recognition in an Open World

[[Project]](https://liuziwei7.github.io/projects/LongTail.html) [[Paper]]() [[Demo]]()      

## Overview
`Open Long-Tailed Recognition (OLTR)` is the author's re-implementation of the long-tail recognizer described in:  
"Large-Scale Long-Tailed Recognition in an Open World"   
[Ziwei Liu](https://liuziwei7.github.io/)<sup>\*</sup>, [Zhongqi Miao](https://github.com/zhmiao)<sup>\*</sup>, [Xiaohang Zhan](https://xiaohangzhan.github.io/), [Jiayun Wang](http://pwang.pw/), [Boqing Gong](http://boqinggong.info/), [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/) (CUHK & UC Berkeley / ICSI & Google)
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2019, **Oral Presentation**

<img src='./assets/intro.png' width=800>

Further information please contact [Zhongqi Miao](zhongqi.miao@berkeley.edu) and [Ziwei Liu](https://liuziwei7.github.io/).

## Requirements
* [PyTorch](https://pytorch.org/)

## Data Preparation

## Download Pre-trained Models
* Caffe pretrained ResNet152 weights can be downloaded from [here](https://drive.google.com/uc?export=download&id=0B7fNdx_jAqhtckNGQ2FLd25fa3c), and save the file to `.logs/caffe_resnet152.pth`

## Getting Started (Training & Testing)

<img src='./assets/pipeline.jpg' width=800>

### Places-LT (*Now only for testing*)
- Stage 1 training:
```
python main.py --config ./config/Places_LT/stage_1_test.py
```
- Stage 2 training:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py True
```
- Close-set testing:
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py --test True
```
- Open-set testing (thresholding)
```
python main.py --config ./config/Places_LT/stage_2_meta_embedding_test.py --test True --test_open True
```

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.
```
@inproceedings{openlongtailrecognition,
  title={Large-Scale Long-Tailed Recognition in an Open World},
  author={Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, Stella X. Yu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
