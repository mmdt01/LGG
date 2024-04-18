"""
Script used to train and test the models: LGGNet: K-FOLD CROSS VALIDATION
Author: Matthys du Toit
Date: 18/04/2024
"""

import mne
from mne import io
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

# cross validation
import torch
from utils import *
from sklearn.model_selection import KFold
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
import pickle

# training
# import torch.nn as nn
CUDA = torch.cuda.is_available()

# from tensorflow import keras
# from EEGModels import EEGNet, DeepConvNet, ShallowConvNet
# from keras import utils as np_utils
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K

##################### Load and epoch data #####################

# define data path
data_path = "subject_4"
file_name = "s4_preprocessed.fif"

# create an event dictionary
event_dict = {
    'motor execution up': 1,
    'motor execution down': 2,
    'visual perception up': 3,
    'visual perception down': 4,
    'imagery up': 5,
    'imagery down': 6,
    'imagery and perception up': 7,
    'imagery and perception down': 8
}

# function for loading the raw data and preparing it into labels and epochs
def load_data(data_path, file_name, event_dict, class_labels, tmin, tmax):
    # load the preprocessed data
    raw = mne.io.read_raw_fif(data_path + "/" + file_name, preload=True)
    # extract the event information from the raw data
    events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)
    # extract epochs from the raw data
    epochs = mne.Epochs(raw, events, event_id=class_labels, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # extract and normalize the labels ensuring they start from 0
    labels = epochs.events[:, -1] - min(epochs.events[:, -1])
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = epochs.get_data()*1e6 # format is in (trials, channels, samples)
    y = labels
    # define the number of kernels, channels and samples
    kernels, chans, samples = 1, X.shape[1], X.shape[2]
    # expand one dimension for deep learning(CNNs)
    X_shaped = np.expand_dims(X, axis=-3)
    # print the shape of the epochs and labels
    print('------------------------------------------------------------------------')
    print("Shape of data: ", X.shape)
    print('------------------------------------------------------------------------')
    print("Shape of data after expanding dimensions: ", X_shaped.shape)
    print('------------------------------------------------------------------------')
    print("Shape of labels: ", y.shape)
    print('------------------------------------------------------------------------')
    print("Labels: ", y)
    print('------------------------------------------------------------------------')
    print("Data and labels prepared!")
    print('------------------------------------------------------------------------')
    # convert labels to one-hot encodings. This is required for the loss function used in the model
    # return the data
    return X_shaped, y, chans, samples

# cross validation
def n_fold_CV(data, label, fold=10, shuffle=True):
    """
    this function achieves n-fold cross-validation
    :param data: epoched eeg data
    :param label: labels for the data
    :param model: model to be trained and tested
    :param fold: how many fold
    """
    # save validation accuracy and f1 score
    va_val = Averager()
    vf_val = Averager()
    preds, acts = [], []
    # define the cross-validation method
    kf = KFold(n_splits=fold, shuffle=shuffle)
    # iterate over the folds
    for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
        # print the fold number
        print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
        # prepare the data
        data_train, label_train, data_test, label_test = prepare_data(idx_train, idx_test, data, label)
        # train the model and get the validation accuracy and f1-score to be used in the second stage
        acc_val, f1_val = first_stage(data=data_train, label=label_train, fold=idx_fold)
        combine_train(data_train, label_train, idx_fold, target_acc=1)
        # test the model
        acc, pred, act = test(data_test, label_test, idx_fold)
        # save the validation accuracy and f1-score
        va_val.add(acc_val)
        vf_val.add(f1_val)
        # save the predictions and actual labels
        preds.extend(pred)
        acts.extend(act)

    # print the mean validation accuracy and f1-score
    print('------------------------------------------------------------------------')
    print('Mean validation accuracy:{}'.format(va_val.item()))
    print('------------------------------------------------------------------------')
    print('Mean validation F1-score:{}'.format(vf_val.item()))
    print('------------------------------------------------------------------------')
    # save the predictions and actual labels
    acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
    print('------------------------------------------------------------------------')
    print('Test accuracy:{}'.format(acc))
    print('------------------------------------------------------------------------')
    print('Test F1-score:{}'.format(f1))
    print('------------------------------------------------------------------------')
    # write the results to a text file
    with open('results.txt', 'w') as f:
        f.write('Mean validation accuracy:{}'.format(va_val.item()))
        f.write('Mean validation F1-score:{}'.format(vf_val.item()))
        f.write('Test accuracy:{}'.format(acc))
        f.write('Test F1-score:{}'.format(f1))




# used in the n_fold_CV function
def prepare_data(idx_train, idx_test, data, label):
    """
    1. get training and testing data according to the index
    2. numpy.array-->torch.tensor
    :param idx_train: index of training data
    :param idx_test: index of testing data
    :param data: (segments, 1, channel, data)
    :param label: (segments,)
    :return: data and label
    """
    data_train = data[idx_train]
    label_train = label[idx_train]
    data_test = data[idx_test]
    label_test = label[idx_test]
    # normalize the data
    data_train, data_test = normalize(data_train, data_test)
    # Prepare the data format for training the model using PyTorch
    data_train = torch.from_numpy(data_train).float()
    label_train = torch.from_numpy(label_train).long()
    data_test = torch.from_numpy(data_test).float()
    label_test = torch.from_numpy(label_test).long()
    # return the prepared data
    return data_train, label_train, data_test, label_test

# used in the prepare_data function
def normalize(train, test):
    """
    this function does standard normalization for EEG channel by channel
    :param train: training data (sample, 1, chan, datapoint)
    :param test: testing data (sample, 1, chan, datapoint)
    :return: normalized training and testing data
    """
    # data: sample x 1 x channel x data
    for channel in range(train.shape[2]):
        mean = np.mean(train[:, :, channel, :])
        std = np.std(train[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
        test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
    return train, test

# used in the n_fold_CV function
def first_stage(data, label, fold):
    """
    this function achieves n-fold-CV to:
        1. select hyper-parameters on training data
        2. get the model for evaluation on testing data
    :param data: (segments, 1, channel, data)
    :param label: (segments,)
    :param fold: which fold the data belongs to
    :return: mean validation accuracy
    """
    # use n-fold-CV to select hyper-parameters on training data
    # save the best performance model and the corresponding acc for the second stage
    # data: trial x 1 x channel x time
    save_path = './save/'
    kf = KFold(n_splits=3, shuffle=True)
    va = Averager()
    vf = Averager()
    va_item = []
    maxAcc = 0.0
    for i, (idx_train, idx_val) in enumerate(kf.split(data)):
        print('Inner 3-fold-CV Fold:{}'.format(i))
        data_train, label_train = data[idx_train], label[idx_train]
        data_val, label_val = data[idx_val], label[idx_val]
        acc_val, F1_val = train(data_train=data_train,
                                label_train=label_train,
                                data_val=data_val,
                                label_val=label_val,
                                # subject=subject,
                                fold=fold)
        va.add(acc_val)
        vf.add(F1_val)
        va_item.append(acc_val)
        if acc_val >= maxAcc:
            maxAcc = acc_val
            # choose the model with higher val acc as the model to second stage
            old_name = osp.join(save_path, 'candidate.pth')
            new_name = osp.join(save_path, 'max-acc.pth')
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(old_name, new_name)
            print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

    mAcc = va.item()
    mF1 = vf.item()
    return mAcc, mF1

# used in the first_stage function
def train(data_train, label_train, data_val, label_val, fold):
    # seed_all(args.random_seed)
    save_name = '_fold' + str(fold)
    # set_up(args)

    # training parameters
    max_epoch = 100
    batch_size = 64
    learning_rate = 1e-3    
    patient = 20
    LS = True
    LS_rate = 0.1
    save_dir = './save/'
    # model parameters
    network = 'LGGNet'
    pool = 16
    pool_step_rate = 0.25
    T = 64
    graph_type = 'gen'
    hidden = 32
    # data parameters
    input_shape = (1, channels, samples)

    # load the data
    train_loader = get_dataloader(data_train, label_train, batch_size, input_shape)
    val_loader = get_dataloader(data_val, label_val, batch_size)

    model = get_model(network, T, pool, pool_step_rate, hidden, graph_type)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # label smoothing
    if LS:
        loss_fn = LabelSmoothing(LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(save_dir, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(save_dir, '{}.pth'.format(name)))

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = patient
    counter = 0

    for epoch in range(1, max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / max_epoch), fold))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(T, pool)
    save_path = osp.join(save_dir, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']

# used in the train function
def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train

# used in the train function
def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def combine_train(data, label, fold, target_acc):
    save_name = '_fold' + str(fold)
    # set_up(args)
    # seed_all(args.random_seed)

    # training parameters
    max_epoch = 100
    max_epoch_cmb = 20
    batch_size = 64
    learning_rate = 1e-3    
    patient = 20
    LS = True
    LS_rate = 0.1
    save_dir = './save/'
    load_path = './save/max-acc.pth'
    # model parameters
    network = 'LGGNet'
    pool = 16
    pool_step_rate = 0.25
    T = 64
    graph_type = 'gen'
    hidden = 32
    # data parameters
    input_shape = (1, channels, samples)

    train_loader = get_dataloader(data, label, batch_size)
    model = get_model(network, T, pool, pool_step_rate, hidden, graph_type)
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*1e-1)

    if LS:
        loss_fn = LabelSmoothing(LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(save_dir, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(save_dir, '{}.pth'.format(name)))

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, max_epoch_cmb + 1):
        loss, pred, act = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
        print('Stage 2 : epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        if acc >= target_acc or epoch == max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')
            # save model here for reproduce
            model_name_reproduce = '_fold' + str(fold) + '.pth'
            data_type = 'model_{}'.format('eeg')
            experiment_setting = 'T_{}_pool_{}'.format(T, pool)
            save_path = osp.join(save_dir, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = osp.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / max_epoch), fold))

    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(T, pool)
    save_path = osp.join(save_dir, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

# used in the n_fold_cv function
def test(data, label, fold):
    # set_up(args)
    # seed_all(args.random_seed)

    # training parameters
    batch_size = 64
    load_path_final = './save/final_model.pth'
    # model parameters
    network = 'LGGNet'
    pool = 16
    pool_step_rate = 0.25
    T = 64
    graph_type = 'gen'
    hidden = 32

    test_loader = get_dataloader(data, label, batch_size)

    model = get_model(network, T, pool, pool_step_rate, hidden, graph_type)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(load_path_final))
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act



################################################################# Run code #############################################################################

# load the data
data, label, channels, samples = load_data(data_path, file_name, event_dict, class_labels=[1,2], tmin=0, tmax=3)

# run the n-fold cross validation
n_fold_CV(data, label, fold=10, shuffle=True)
