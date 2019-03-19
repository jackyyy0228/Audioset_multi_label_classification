import torch 
import torch.utils.data as torch_data
import os
import numpy as np
import pickle as pkl

def get_data(config):
    #X_train, Y_train = pkl.load(open(os.path.join(config.data_path,'train_data.pkl'),'rb'))
    X_test, Y_test = pkl.load(open(os.path.join(config.data_path,'test_data.pkl'),'rb'))
    #X_train = (np.float32(X_train) - 128.) / 128.
    X_test = (np.float32(X_test) - 128.) / 128.
    X_train, Y_train = X_test, Y_test 
    return X_train, Y_train, X_test, Y_test

''' 
def split_data(X, Y, num_valid, random_seed = 777):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    valid_idxs, train_idxs = idxs[:num_valid], idxs[num_valid:]
    Y_valid, Y_train = [], []
    for idx in valid_idxs:
        Y_valid.append(Y[idx])
    for idx in train_idxs:
        Y_train.append(Y[idx])
    return X[valid_idxs], Y_valid, X[train_idxs], Y_train
'''
