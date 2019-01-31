import numpy as np
import matplotlib.pyplot as plt
import torch

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def class_num_count (train_data):

    input_samples = np.array(train_data.dataset.samples)

    labels = input_samples[:, 1].astype(int)

    class_data_num = []

    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
        
    return class_data_num

def shot_acc (class_count, class_correct, class_total, 
              many_shot_thr=100, low_shot_thr=20):
    
    many_shot = []
    median_shot = []
    low_shot = []

    for i in range(len(class_count)):
        if class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / class_total[i]).item())
        elif class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / class_total[i]).item())
        else:
            median_shot.append((class_correct[i] / class_total[i]).item())
            
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def dataset_dist (in_loader):

    """Example, dataset_dist(data['train'][0])"""
    
    label_list = np.array([x[1] for x in in_loader.dataset.samples])
    total_num = len(data_list)

    distribution = []
    for l in np.unique(label_list):
        distribution.append((l, len(label_list[label_list == l])/total_num))
        
    return distribution

def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        
def init_weights(model, weights_path, caffe=False, classifier=False):
    
    """Initialize weights"""

    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    
    weights = torch.load(weights_path)
    
    if not classifier:

        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                       for k in model.state_dict()}

    else:
        
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k] 
                   for k in model.state_dict()}

    model.load_state_dict(weights)
    
    return model

def F_measure(preds, labels):

    true_pos = 0.
    false_pos = 0.
    false_neg = 0.

    for i in range(len(labels)):

        true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
        false_pos += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
        false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    return 2 * ((precision * recall) / (precision + recall + 1e-12))
