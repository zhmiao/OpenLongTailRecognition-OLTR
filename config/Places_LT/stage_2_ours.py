# Testing configurations
config = {}

training_opt = {}
training_opt['dataset'] = 'places365_lt'
training_opt['log_dir'] = './logs/Places_LT/relation'
training_opt['num_classes'] = 365
training_opt['batch_size'] = 256
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 30
training_opt['display_step'] = 1
training_opt['feature_dim'] = 512
training_opt['open_threshold'] = 0.1
training_opt['sampler'] = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py',
                           'num_samples_cls': 4}
training_opt['scheduler_params'] = {'step_size':10, 'gamma':0.1}
config['training_opt'] = training_opt

networks = {}
feature_param = {'pretrain': True, 'weights': 'places_pre', 'use_fc': True, 'dropout': None}
feature_optim_param = {'lr': 0.01, 'momentum':0.9, 'weight_decay':0.0005}
networks['feat_model'] = {'def_file': './models/ResNet152Feature_ours.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': True}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'], 'stage1_weights': 'places_pre'}
classifier_optim_param = {'lr': 0.1, 'momentum':0.9, 'weight_decay':0.0005}
networks['classifier'] = {'def_file': './models/RelationClassifier_ours_update_v3.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
feat_loss_param = {'feat_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes']}
feat_loss_optim_param = {'lr': 0.01, 'momentum':0.9, 'weight_decay':0.0005}
criterions['FeatureLoss'] = {'def_file': './loss/CenterLoss_ours.py', 'loss_params': feat_loss_param,
                             'optim_params': feat_loss_optim_param, 'weight': 0.01}
config['criterions'] = criterions

relations = {}
relations['centers'] = True
relations['attention'] = True
relations['init_centers'] = True
config['relations'] = relations

