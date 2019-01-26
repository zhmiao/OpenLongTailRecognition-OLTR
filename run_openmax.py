import os
import numpy as np
import scipy.spatial.distance as spd
import libmr
from tqdm import tqdm
import time
from multiprocessing import Pool
from functools import partial
from utils import *

def softmax(logits):
    if logits.ndim > 1:
        return np.exp(logits) / np.expand_dims(np.exp(logits).sum(axis=1), axis=1)
    else:
        return np.exp(logits) / np.exp(logits).sum()

def mean_and_distance(logits_train, labels_train, num_classes):

    # Calculate softmax for train logits
    softmax_train = softmax(logits_train)

    # Calculate predictions from softmax
    preds_train = softmax_train.argmax(axis=1)

    # Find the indexes of correct predictions
    correct_idx_train = (preds_train == labels_train)

    # Extract correctly predicted logits and corresponding labels
    correct_logits_train = logits_train[correct_idx_train]
    correct_labels_train = labels_train[correct_idx_train]

    # Separate the logits by class
    sep_correct_logits_train = {}
    for cat in range(num_classes):
        sep_correct_logits_train[cat] = correct_logits_train[correct_labels_train == cat]\
                                                    if cat in correct_labels_train else None

    # Calculate mean logits and corresponding euclidean-cosine distances
    correct_mean_train = {}
    correct_distances_train = {}

    for cat in range(num_classes):
            # If current category is never correctly predicted, then set to None
        if not sep_correct_logits_train[cat] is None:
            class_logits = sep_correct_logits_train[cat]
            mean = class_logits.mean(axis=0)
            eu_dist = np.array([spd.euclidean(mean, l) for l in class_logits])
            cos_dist = np.array([spd.cosine(mean, l) for l in class_logits])
            eucos_dist = eu_dist / 200. + cos_dist
            correct_mean_train[cat] = mean
            correct_distances_train[cat] = eucos_dist
        else:
            correct_mean_train[cat] = None
            correct_distances_train[cat] = None

    return correct_mean_train, correct_distances_train

def fit_weibull (correct_mean_train, correct_distances_train, num_classes, tailsize):

    # Now, fit weibull for each class. 
    # If class is never correctly predicted, the model will be None
    weibull_model = {}

    for cat in range(num_classes):
    
        if not correct_mean_train[cat] is None:
           
            mean = correct_mean_train[cat]
            distances = correct_distances_train[cat]

            weibull_model_cat = {}

            weibull_model_cat['distances'] = distances
            weibull_model_cat['mean_vec'] = mean

            mr = libmr.MR()
            tailtofit = sorted(distances)[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))

            weibull_model_cat['weibull_model'] = mr

            weibull_model[cat] = weibull_model_cat
           
        else:
           
            weibull_model[cat] = None

    return weibull_model

def compute_openmax(weibull_model, alpharank, num_classes, query_logits):

    # Calculate softmax probabilities of query logits
    probs = softmax(query_logits)

    # Calculate ranked alpha
    ranked_list = probs.argsort()[::-1]
    alpha_weights = [(alpharank + 1 - i) / float(alpharank) 
                  for i in range(1, alpharank+1)]
    ranked_alpha = np.zeros(num_classes)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Calculate openmax logits for query logits
    openmax_logits = []
    openmax_logits_unknown = []

    # Loop over classes
    for cat in range(num_classes):
    
        if not weibull_model[cat] is None:
            
            weibull_model_cat = weibull_model[cat]

            mean_vec = weibull_model_cat['mean_vec']

            # Calculate euclidean-cosine distances of query logits to mean logits
            # of current class
            eucos_distance = spd.euclidean(mean_vec, query_logits) / 200.\
                           + spd.cosine(mean_vec, query_logits)

            # Calculate wscore based on the distance
            wscore = weibull_model_cat['weibull_model'].w_score(eucos_distance)

            # For stability
            wscore = wscore if wscore > 0 else 0.

            # Calculate modified logits of current class
            modified_logit_cat = query_logits[cat] * ( 1 - wscore * ranked_alpha[cat])

            # Append the results
            openmax_logits += [modified_logit_cat]
            openmax_logits_unknown += [query_logits[cat] - modified_logit_cat]
           
        else:
            # If current class is never correctly predicted, 
            # simply use origin logits
            openmax_logits += [query_logits[cat]]
            openmax_logits_unknown += [query_logits[cat]]

    openmax_logits = np.array(openmax_logits)
    openmax_logits_unknown = np.array(openmax_logits_unknown)

    # Calculate openmax probabilities
    # First, calculate query scores
    query_scores = np.exp(openmax_logits)

    # Then, calculate total denominator
    total_denominator = np.sum(np.exp(openmax_logits)) + np.exp(np.sum(openmax_logits_unknown))

    # Openmax score
    openmax_scores = query_scores / total_denominator

    # Unknown score
    unknowns = np.exp(np.sum(openmax_logits_unknown)) / total_denominator

    # Combine the scores
    modified_scores =  openmax_scores.tolist() + [unknowns]

    return modified_scores

def compute_acc (openmax_probs, labels_test, labels_train, num_classes):

    # First, split classes into many-shot, median-shot, and low-shot
    many_shot_cat = []
    median_shot_cat = []
    low_shot_cat = []
    for cat in range(num_classes):
        
        data_num = len(labels_train[labels_train == cat])
        
        if data_num > 100:
            many_shot_cat.append(cat)
        elif data_num < 20:
            low_shot_cat.append(cat)
        else:
            median_shot_cat.append(cat)

    # Calculate predictions using openmax probabilities, set open set label to -1
    openmax_preds = openmax_probs.argmax(axis=1)
    openmax_preds[openmax_preds == num_classes] = -1

    # Calculate close set average accuracy
    idx = labels_test != -1
    avg_acc_close = (openmax_preds[idx] == labels_test[idx]).sum() / len(labels_test[idx])

    # Calculate close set many-shot accuracy
    idx = np.isin(labels_test, many_shot_cat)
    many_shot_acc_close = (openmax_preds[idx] == labels_test[idx]).sum() / len(labels_test[idx])

    # Calculate close set median-shot accuracy
    idx = np.isin(labels_test, median_shot_cat)
    median_shot_acc_close = (openmax_preds[idx] == labels_test[idx]).sum() / len(labels_test[idx])

    # Calculate close set low-shot accuracy
    idx = np.isin(labels_test, low_shot_cat)
    low_shot_acc_close = (openmax_preds[idx] == labels_test[idx]).sum() / len(labels_test[idx])

    # Calculate open set f-measurement
    f_measurement = F_measure(openmax_preds, labels_test)

    return avg_acc_close, many_shot_acc_close, median_shot_acc_close, low_shot_acc_close, f_measurement


def openmax_results (log_dir, num_classes, tailsize=5, alpharank=10, workers=10):

    print('###############################################', '\n')
    print('Tailsize: %d' % tailsize)
    print('Alpharank: %d' % alpharank)

    # Load logits of test and training data
    print('Loading saved logits.')
    test_npz = np.load(os.path.join(log_dir, 'logits_out_test.npz'))
    train_npz = np.load(os.path.join(log_dir, 'logits_out_train.npz'))
    logits_train = train_npz['logits']
    labels_train = train_npz['labels']
    logits_test = test_npz['logits']
    labels_test = test_npz['labels']

    # First, calculate mean logits and corresponding distances
    # of each logit-vector to the means
    print('Calculating mean logits and corresponding distances.')
    correct_mean_train, correct_distances_train = mean_and_distance(logits_train, labels_train, num_classes)

    # Second, fit weibull model
    print('Fitting Weibull model.')
    weibull_model = fit_weibull(correct_mean_train, correct_distances_train, num_classes, tailsize)
    time.sleep(0.01)

    # Calculate openmax and softmax for train logits of test data
    print('Calculating openmax probabilities for testing data.')

    print('Using %d workers.' % workers)

    with Pool(workers) as pool:

        func = partial(compute_openmax, weibull_model, alpharank, num_classes)

        openmax_probs = list(tqdm(pool.imap(func, logits_test, chunksize=24),
                             total=len(logits_test)))

    openmax_probs = np.array(openmax_probs)

    # return openmax_probs, softmax(logits_test)

    avg_acc_close,\
    many_shot_acc_close,\
    median_shot_acc_close,\
    low_shot_acc_close,\
    f_measurement = compute_acc(openmax_probs, labels_test, labels_train, num_classes)

    print_str =  ['Close set accuracies:',
                 '\n',
                 'Evaluation_accuracy_micro_top1: %.3f' 
                 % (avg_acc_close),
                 'Evaluation_accuracy_macro_top1: %.3f' 
                 % (avg_acc_close),
                 'Many_shot_accuracy_top1: %.3f' 
                 % (many_shot_acc_close),
                 'Median_shot_accuracy_top1: %.3f' 
                 % (median_shot_acc_close),
                 'Low_shot_accuracy_top1: %.3f' 
                 % (low_shot_acc_close),
                 '\n',
                 '\n',
                 'Open set measurement:',
                 '\n',
                 'F-Measure: %.3f'
                 % (f_measurement)]

    print(*print_str)

    # print('\n', '###', '\n')
    # print('F-measurement: %.4f' % f_measurement)
    # print('\n', '###', '\n')

    return f_measurement, avg_acc_close, many_shot_acc_close, median_shot_acc_close, low_shot_acc_close

def grid_search(workers):

    log_dir = './log/imagenet_test/plain/'
    num_classes = 1000

    gridsearch_logfile = './openmax_gridsearch_log.txt'

    if os.path.isfile(gridsearch_logfile):
        os.remove(gridsearch_logfile)

    best_fm = 0.
    best_tail = 0
    best_alpha = 0

    alpharank_list = [0, 1, 2]

    tail_list = [0, 1, 2]

    for alpha in alpharank_list:

        for tail in tail_list:

            f_measurement, _, _, _, _ = openmax_results (log_dir, num_classes, 
                                              tailsize=tail, alpharank=alpha, 
                                              workers=workers)

            log_string = ['\n',
                        'Tailsize: %d' % tail,
                        'Alpharank: %d' % alpha,
                        'F-measurement: %.4f' % f_measurement,
                        '\n']

            print_write(log_string, gridsearch_logfile)

            if f_measurement > best_fm:
                best_fm = f_measurement
                best_tail = tail
                best_alpha = alpha

    log_string = ['\n',
                'Best Values: ',
                '\n',
                'Tailsize: %d' % best_tail,
                'Alpharank: %d' % best_alpha,
                'F-measurement: %.4f' % best_fm,
                '\n']

    print_write(log_string, gridsearch_logfile)

if __name__ == '__main__':
    grid_search(8)


