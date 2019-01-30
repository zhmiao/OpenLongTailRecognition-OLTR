import os
import imp
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import numpy as np
import warnings
import pdb

class model ():
    
    def __init__(self, config, data, use_step=False, debug_mode=False, 
        test=False, test_open=False):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.relations = self.config['relations']
        self.data = data
        self.use_step = use_step
        self.test_mode = test
        self.test_open = test_open
        
        if 'train' in self.data.keys():
            self.class_count = class_num_count (self.data['train'][0])
        else:
            warnings.warn('Input data do not have training set. Class count can not be calculated',
                          UserWarning)
            self.class_count = None

        # If using steps for training, we need to calculate training steps 
        # for each epoch based on actual number of training data instead of 
        # oversampled data number 
        if self.use_step and test == False:
            print('Using steps for training.')
            self.training_data_num = self.data['train'][1]
            self.total_steps = int(self.training_data_num \
                                 * self.training_opt['num_epochs'] \
                                 / self.training_opt['batch_size'])
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])
        
        # If not under debug mode, we need to initialize the models,
        # criterions, and centers if needed.
        if not debug_mode:
            self.scheduler_params = self.training_opt['scheduler_params']
            self.init_models()
            if not self.test_mode:
                self.init_criterions()
                if self.relations['init_centers']:
                    self.init_centers()

        if test_open:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])


    def init_models(self):
        
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optimizer_params = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = val['params'].values()

            self.networks[key] = imp.load_source('', def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights.')
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
            self.criterions[key] = imp.load_source('', def_file).create_loss(*loss_args).to(self.device)
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

    def init_centers(self):
        initial_centers = torch.zeros(self.training_opt['num_classes'],
                                      self.training_opt['feature_dim']).cuda()

        print('Calculating initial centers.')

        # Calculate initial centers only on un-upsampled training data.
        with torch.set_grad_enabled(False):

            self.networks['feat_model'].eval()

            for inputs, labels, _ in tqdm(self.data['train_plain'][0]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                features, _, _ = self.networks['feat_model'](inputs, labels, self.class_count)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    initial_centers[label] += features[i]

        # Average summed features with class count
        initial_centers /= torch.tensor(self.class_count).float().unsqueeze(1).cuda()

        # Assign initial centers
        self.criterions['FeatureLoss'].centers.data = initial_centers

    def output_logits(self):

        for phase in ['train', 'test']:
        # for phase in ['test']:

            self.logits_out = np.empty((0, self.training_opt['num_classes']))
            self.labels_out = np.empty(0, int)

            with torch.set_grad_enabled(False):

                print('Calculating logits for dataset %s' % (phase))

                for model in self.networks.values():
                    model.eval()

                for inputs, labels, _ in tqdm(self.data[phase][0]):

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.batch_run(inputs, labels, output_logits=True, phase='test')

            print('Saving logits to %s' %(self.training_opt['log_dir']+'logits_out_%s.npz' % phase))
            np.savez(self.training_opt['log_dir']+'logits_out_%s.npz' % phase,
                     logits=self.logits_out, 
                     labels=self.labels_out,)

            
    def batch_run (self, inputs, labels, loss=True, top5=False, centers=True, attention=True, phase='train', output_logits=False):

        '''
        This is a general single batch running function. 
        '''
        
        # Calculate Features with loss attention if exist
        self.features, self.loss_attention, self.feature_maps = self.networks['feat_model'](inputs, labels, self.class_count)
        # self.features, self.loss_attention = self.networks['feat_model'](inputs, labels, self.class_count)
        
        # During training and validation, calculate centers if needed to 
        if phase != 'test':
            if centers and 'FeatureLoss' in self.criterions.keys():
                self.centers = self.criterions['FeatureLoss'].centers.data
            else:
                self.centers = None

        # Calculate logits with classifier
        self.logits, self.loss_relation, self.slow_fast_feature = self.networks['classifier'](self.features, labels, self.centers, self.class_count)
        
        if output_logits:

            self.logits_out = np.append(self.logits_out, self.logits.cpu().numpy(), axis=0)
            self.labels_out = np.append(self.labels_out, labels.cpu().numpy(), axis=0)

        else:

            # Calculate Top 1 prediction, 
            probs, self.preds_top1 = F.softmax(self.logits.detach(), dim=1).max(dim=1)
            
            # During open test, set -1 to data with highest probs under threshold
            # setup in the config files
            if self.test_open:
                self.preds_top1[probs < self.training_opt['open_threshold']] = -1

            # If needed, calculate Top 5 prediction
            if top5:
                _, self.preds_top5 = self.logits.detach().topk(5, 1)

            # If needed, calculate losses
            if loss:
                # First, apply performance loss
                self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                            * self.criterion_weights['PerformanceLoss']

                # Add performance loss to total loss
                self.loss = self.loss_perf

                # Apply loss on features if set up
                if 'FeatureLoss' in self.criterions.keys():
                    self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
                    self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
                    # Add feature loss to total loss
                    self.loss += self.loss_feat

                # If needed, obtain attention loss from feature model
                # if attention:
                if self.loss_attention:
                    # self.loss_attention = self.networks['feat_model'].loss_attention
                    # self.loss_relation = self.networks['classifier'].loss_relation
                    # Add attention loss to total loss
                    self.loss += self.loss_attention

                if self.loss_relation:
                    self.loss += self.loss_relation

            if phase == 'train':

                # pdb.set_trace()

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

    def run(self, mode='train', calc_conf_mat=False):

        # running mode can only be train or test
        assert mode in ['train', 'test']
        
        if mode == 'train':
            # When training the network
            # Set up log file
            log_file = os.path.join(self.training_opt['log_dir'], 
                                    'log.txt' if not self.use_step else 'log_step.txt')
            if os.path.isfile(log_file):
                os.remove(log_file) 
            
            # Initialize best model
            best_model_weights = {}
            best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
            best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            best_acc = 0.0
            best_epoch = 0

            # Initialize training step
            training_step = 1
            # Ending epoch is training epoch
            end_epoch = self.training_opt['num_epochs']
            # Phase list include both training and evaluation
            phase_list = ['train', 'val']
            # Swith for loop break under step training mode
            need_break = True if self.use_step else False

        else:
            # When testing the network
            # Only need one epoch for testing
            end_epoch = 1
            # Only need evaluation for testing
            phase_list = ['test']
            # No break under evaluation mode
            need_break = False

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            # Loop over training phase and validation phase
            for phase in phase_list:
                
                torch.cuda.empty_cache()
                
                # test flag
                # phase = 'val'
                
                # Set model modes and set scheduler
                if phase == 'train':
                    # In training, step optimizer scheduler and set model to train() 
                    self.model_optimizer_scheduler.step()
                    if self.criterion_optimizer:
                        self.criterion_optimizer_scheduler.step()
                    for model in self.networks.values():
                        model.train()
                else:
                    # In validation or testing mode, set model to eval() and initialize running loss/correct
                    for model in self.networks.values():
                        model.eval()
                    # Initialize running loss 
                    # self.running_loss = 0.0
                    # Initialize top 1 running correct
                    self.class_total = torch.tensor([0. for i in range(self.training_opt['num_classes'])],
                                                    requires_grad=False)
                    self.running_correct_total_top1 = 0.
                    self.class_correct_top1 = torch.tensor([0. for i in range(self.training_opt['num_classes'])],
                                                           requires_grad=False)

                    # Initialize top 5 running correct or f measure
                    if self.test_open:
                        self.running_f_measure = 0.
                    else:
                        self.running_correct_total_top5 = 0
                        self.class_correct_top5 = torch.tensor([0. for i in range(self.training_opt['num_classes'])],
                                                               requires_grad=False)
                        

                    # If under test mode, initialize confusion matrix
                    if calc_conf_mat:
                        self.conf_mat = torch.tensor([[0. for i in range(self.training_opt['num_classes'])]
                                                          for j in range(self.training_opt['num_classes'])])

                # Iterate over dataset
                for self.inputs, self.labels, self.paths in (self.data[phase][0] \
                                                             if phase == 'train' \
                                                             else tqdm(self.data[phase][0])):
                    inputs, labels = self.inputs.to(self.device), self.labels.to(self.device)

                    if need_break \
                        and phase == 'train' \
                        and training_step % self.epoch_steps == 0:
                        break

                    # If on training phase, enable gradients
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        if phase == 'train':
                            
                            # Under step training mode, every time enters training loop, break switch set to True
                            need_break = True if self.use_step else False
                            # If training, forward with loss, and no top 5 accuracy calculation
                            self.batch_run(inputs, labels, loss=True, top5=False,
                                         centers=self.relations['centers'],
                                         attention=self.relations['attention'],
                                         phase=phase)

                            # After back-propagation, training step + 1
                            training_step += 1

                            # Output minibatch training results
                            if training_step % self.training_opt['display_step'] == 0:

                                minibatch_loss_feat = self.loss_feat.item() \
                                    if 'FeatureLoss' in self.criterions.keys() else None
                                minibatch_loss_perf = self.loss_perf.item()
                                minibatch_acc = (self.preds_top1 == labels).sum().item() \
                                                / self.training_opt['batch_size']

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
                                print_write(print_str, log_file)

                        else:
                            # In evaluation loop, break switch set to False
                            need_break = False
                            # In validation or testing, forward with top 5 accuracy
                            self.batch_run(inputs, labels, loss=False, top5=False if self.test_open else True,
                                         centers=self.relations['centers'],
                                         attention=self.relations['attention'],
                                         phase=phase)

                            # Record top 1 running correct
                            # Only count close set data here with true labels != -1
                            correct_tensor_top1 = (self.preds_top1[labels != -1] == labels[labels != -1])
                            self.running_correct_total_top1 += correct_tensor_top1.sum().item()
                            # Record top 5 running correct if not using open set 
                            # otherwise record f-measurement
                            if self.test_open:
                                self.running_f_measure += F_measure(self.preds_top1, labels) * inputs.shape[0]
                            else:
                                correct_tensor_top5 = torch.tensor([1 if labels[i] in self.preds_top5[i] else 0 
                                                                    for i in range(len(labels))])
                                self.running_correct_total_top5 += correct_tensor_top5.sum().item()
                                
                            # Record correct tensors per class
                            # also, not counting true labels == -1
                            for i in range(len(labels[labels != -1])):
                                label = labels[labels != -1][i]
                                self.class_total[label] += 1
                                # Top 1
                                self.class_correct_top1[label] += correct_tensor_top1[i].item()
                                # Top 5
                                if not self.test_open:
                                    self.class_correct_top5[label] += correct_tensor_top5[i].item()
                                # Conf Mat
                                if calc_conf_mat:
                                    pred_top1 = self.preds_top1[i]
                                    self.conf_mat[label, pred_top1] += 1

                # If under validation or testing, calculate accuracy on the validation set
                if phase != 'train':

                    # Top 1
                    self.acc_mic_top1 = self.running_correct_total_top1 / self.class_total.sum().item()
                    self.acc_mac_top1 = (self.class_correct_top1 / self.class_total).mean().item()
                    self.many_acc_top1, \
                    self.median_acc_top1, \
                    self.low_acc_top1 = shot_acc(self.class_count, 
                                                     self.class_correct_top1, 
                                                     self.class_total)

                    # Top 5 or f-measure
                    if self.test_open:
                        self.f_measure = self.running_f_measure / self.data[phase][1]
                    else:
                        self.acc_mic_top5 = self.running_correct_total_top5 / self.class_total.sum().item()
                        self.acc_mac_top5 = (self.class_correct_top5 / self.class_total).mean().item()
                        self.many_acc_top5, \
                        self.median_acc_top5, \
                        self.low_acc_top5 = shot_acc(self.class_count, 
                                                         self.class_correct_top5,
                                                         self.class_total)

                    # Confusion matrix
                    if calc_conf_mat:
                       self.conf_mat = self.conf_mat/(self.class_total.reshape((len(self.class_total), 1)))
                       np.save(self.training_opt['log_dir']+'conf_mat.npy', self.conf_mat.numpy())
                       print('Confusion matrix saved to %s' % (self.training_opt['log_dir']+'conf_mat.npy'))

                    # Print out accuracy

                    # Top-1 accuracy and additional string
                    print_str = ['Phase: %s' 
                                 % (phase),
                                 'Epoch: [%d/%d]' 
                                 % (epoch, self.training_opt['num_epochs']),
                                 '\n\n',
                                 'Close set accuracies:',
                                 '\n',
                                 'Evaluation_accuracy_micro_top1: %.3f' 
                                 % (self.acc_mic_top1),
                                 'Evaluation_accuracy_macro_top1: %.3f' 
                                 % (self.acc_mac_top1),
                                 'Many_shot_accuracy_top1: %.3f' 
                                 % (self.many_acc_top1),
                                 'Median_shot_accuracy_top1: %.3f' 
                                 % (self.median_acc_top1),
                                 'Low_shot_accuracy_top1: %.3f' 
                                 % (self.low_acc_top1),
                                 '\n']

                    # Additional accuracy string than top-1 accuracy (top 5 or f measure)
                    add_str = ['\n',
                               'Open set measurement:',
                               '\n',
                               'F-Measure: %.3f'
                               % (self.f_measure)] if self.test_open else ['Evaluation_accuracy_micro_top5: %.3f' 
                                                                            % (self.acc_mic_top5),
                                                                            'Evaluation_accuracy_macro_top5: %.3f' 
                                                                            % (self.acc_mac_top5),
                                                                            'Many_shot_accuracy_top5: %.3f' 
                                                                            % (self.many_acc_top5),
                                                                            'Median_shot_accuracy_top5: %.3f' 
                                                                            % (self.median_acc_top5),
                                                                            'Low_shot_accuracy_top5: %.3f' 
                                                                            % (self.low_acc_top5),]

                    print_str += add_str

                    if phase != 'test':
                        print_write(print_str, log_file)
                    else:
                        print(*print_str)

                    # pdb.set_trace()
                    
                    # Under validation, the best model need to be updated
                    if phase == 'val' and self.acc_mac_top1 > best_acc:
                        best_epoch = epoch
                        best_acc = self.acc_mac_top1
                        best_centers = self.centers
                        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

                        # Save model every 5 epochs
                        # if epoch % 5 == 0:
                        #     self.save_model(epoch, centers=self.centers)

        if mode == 'train':
            print()
            print('Training Complete.')
            print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
            print_write(print_str, log_file)
            # Save the best model and best centers if calculated
            self.save_model(epoch, best_epoch, best_model_weights, best_acc, centers=best_centers)
    
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
            model.load_state_dict(model_state[key])
        
    
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
            
