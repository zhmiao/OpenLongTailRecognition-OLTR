import os
import imp
import argparse
import pprint
from data import dataloader
from run_networks import model
from run_openmax import openmax_results
import warnings

# ================
# LOAD CONFIGURATIONS

data_root = {'imagenet_lt': '/home/public/dataset/imagenet_LT/',
             'sun397_lt': '/home/public/dataset/sun397_LT/',
             'places365_lt': '/home/public/dataset/Places365_LT/'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test', default=False)
parser.add_argument('--test_epoch', default=None, type=int)
parser.add_argument('--test_open', default=False)
parser.add_argument('--use_step', default=False)
parser.add_argument('--output_logits', default=False)
parser.add_argument('--output_conf_mat', default=False)
args = parser.parse_args()

test_mode = args.test
test_epoch = args.test_epoch
test_open = args.test_open
output_logits = args.output_logits
output_conf_mat = args.output_conf_mat

use_step = args.use_step

config = imp.load_source("", args.config).config
training_opt = config['training_opt']
relatin_opt = config['relations']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset])
pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': imp.load_source('', sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {x: dataloader.load_data(data_root=data_root[dataset], dataset=x, 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic, num_workers=training_opt['num_workers'])
            for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centers'] else ['train', 'val'])}

    training_model = model(config, data, use_step=use_step, test=False)

    training_model.run(mode='train')

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    data = {x: dataloader.load_data(data_root=data_root[dataset], dataset=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=not output_logits)
            for x in ['train', 'test']}

    
    training_model = model(config, data, use_step=use_step, test=True,
                           test_open=test_open)
    training_model.load_model(epoch=test_epoch)
    
    if not output_logits:
        training_model.run(mode='test', calc_conf_mat=output_conf_mat)
    else:
        training_model.output_logits()
        

