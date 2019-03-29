import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
import warnings
import pdb

class model ():
    
    def __init__(self, config, data, use_step=False, test=False):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.relations = self.config['relations']
        self.data = data
        self.use_step = use_step
        self.test_mode = test

        # If using steps for training, we need to calculate training steps 
        # for each epoch based on actual number of training data instead of 
        # oversampled data number 
        if self.use_step and test == False:
            print('Using steps for training.')
            self.training_data_num = self.data['train'][1]
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])
        
        # If not under debug mode, we need to initialize the models,
        # criterions, and centers if needed.
        self.scheduler_params = self.training_opt['scheduler_params']
        self.init_models()
        if not self.test_mode:
            self.init_criterions()
            if self.relations['init_centers']:
                self.criterions['FeatureLoss'].centers.data = self.centers_cal(self.data['train_plain'][0])
            
        # Set up log file
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
        
    def init_models(self, optimizer=True):
        
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optimizer_params = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = val['params'].values()

            # pdb.set_trace()

            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optimizer_params.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})

        # Initialize model optimizer and scheduler
        print('Initializing model optimizer.')
        self.model_optimizer = optim.SGD(self.model_optimizer_params)
        self.model_optimizer_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer,
                                                                   step_size=self.scheduler_params['step_size'],
                                                                   gamma=self.scheduler_params['gamma'])

    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer = optim.SGD(params=self.criterions[key].parameters(),
                                                 lr=optim_params['lr'],
                                                 momentum=optim_params['momentum'],
                                                 weight_decay=optim_params['weight_decay'])

                self.criterion_optimizer_scheduler = optim.lr_scheduler.StepLR(self.criterion_optimizer,
                                                                               step_size=self.scheduler_params['step_size'],
                                                                               gamma=self.scheduler_params['gamma'])
            else:
                self.criterion_optimizer = None

    def batch_forward (self, inputs, labels=None, centers=False, feature_ext=False, phase='train'):

        '''
        This is a general single batch running function. 
        '''
        
        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)
        
        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centers if needed to 
            if phase != 'test':
                if centers and 'FeatureLoss' in self.criterion.keys():
                    self.centers = self.criterion['FeatureLoss'].centers.data
                else:
                    self.centers = None

            # Calculate logits with classifier
            self.logits, self.slow_fast_feature = self.networks['classifier'](self.features, self.centers)

    def batch_backward(self):

        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()

        # Back-propagation from loss outputs
        self.loss.backward()

        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):

        # First, apply performance loss
        self.loss_perf = self.criterion['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # Add performance loss to total loss
        self.loss = self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterion.keys():
            self.loss_feat = self.criterion['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def train(self):

        # When training the network
        
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        # Initialize training step
        # training_step = 1
        # Ending epoch is training epoch
        end_epoch = self.training_opt['num_epochs']

        for model in self.networks.values():
            model.train()

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
                
            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            for step, (inputs, labels, _) in enumerate(self.data['train'][0]):

                # Break when step equal to epoch step
                if self.use_step and step == self.epoch_steps:
                    break

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, 
                                       centers=self.relations['centers'],
                                       phase='train')
                    self.batch_loss(labels)
                    self.batch_backward()

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterion.keys() else None
                        minibatch_loss_perf = self.loss_perf.item()
                        minibatch_acc = mic_acc_cal(self.logits, labels)

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (training_step),
                                     'Minibatch_loss_feature: %.3f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f' 
                                     % (minibatch_loss_perf),
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc)]
                        print_write(print_str, self.log_file)

            # After every epoch, validation
            self.eval(phase='val')

            # Under validation, the best model need to be updated
            if self.eval_acc_mic > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic
                best_centers = self.centers
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        print()
        print('Training Complete.')

        if save_best:
            print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
            print_write(print_str, self.log_file)
            # Save the best model and best centers if calculated
            self.save_model(epoch, best_epoch, best_model_weights, best_acc, centers=best_centers)
                
        print('Done')

    def eval(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # if test_open:
        #     print('Under openset test mode. Open threshold is %.1f' 
        #           % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        # self.total_paths = np.empty(0)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase][0]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centers=self.relations['centers'],
                                   phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                # self.total_paths = np.concatenate((self.total_paths, paths))

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(self.total_logits[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(self.total_logits, self.total_labels, openset=openset)
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1 = shot_acc(self.total_logits[self.total_labels != -1],
                                     self.total_labels[self.total_labels != -1], 
                                     self.data['train'][0])
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)
            
    def centers_cal(self, data):

        centers = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centers.')

        for model in self.networks.values():
            model.eval()

        # Calculate initial centers only on training data.
        with torch.set_grad_enabled(False):
            
            for inputs, labels, _ in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centers[label] += self.features[i]

        # Average summed features with class count
        centers /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centers

    def load_model(self, epoch=None):

        if not epoch:
            
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                     'final_model_checkpoint.pth' \
                                     if not self.use_step else 'final_model_checkpoint_step.pth')
            
            print('Validation on the best model.')
            print('Loading model from %s' % (model_dir))
            
            checkpoint = torch.load(model_dir)          
            model_state = checkpoint['state_dict_best']
            
            self.centers = checkpoint['centers'] if 'centers' in checkpoint else None
            
        else:
            
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                     ('epoch_%s_checkpoint.pth' % epoch) \
                                     if not self.use_step else ('epoch_%s_checkpoint_step.pth' % epoch))
            
            print('Validation on the model of epoch %d' % epoch)            
            print('Loading model from %s' % (model_dir))
            
            checkpoint = torch.load(model_dir)            
            model_state = checkpoint['state_dict']
            
            self.centers = checkpoint['centers'] if 'centers' in checkpoint else None
        
        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
        
    
    def save_model(self, epoch, best_epoch=None, best_model_weights=None, best_acc=None, centers=None):
        
        #pdb.set_trace()
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'state_dict': {'feat_model': self.networks['feat_model'].state_dict(),
                               'classifier': self.networks['classifier'].state_dict()},
                'best_acc': best_acc,
                'model_optimizer': self.model_optimizer.state_dict(),
                'criterion_optimizer': self.criterion_optimizer.state_dict() if self.criterion_optimizer else None,
                'centers': centers}

        if best_epoch:
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                     'final_model_checkpoint.pth' \
                                     if not self.use_step else 'final_model_checkpoint_step.pth')
            torch.save(model_states, model_dir)
        else:
            model_dir = os.path.join(self.training_opt['log_dir'], 
                                     ('epoch_%s_checkpoint.pth' % epoch) \
                                     if not self.use_step else ('epoch_%s_checkpoint_step.pth' % epoch))
            torch.save(model_states, model_dir)
            
