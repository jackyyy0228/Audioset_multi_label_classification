import os, sys
import numpy as np
import h5py 
import pickle

def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
        video_id_list = hf['video_id_list'][:].tolist()
    x = np.float32(x)
    y_list = [] 
    for label in y:
        y_list.append(list(np.where(label == 1)[0]))

    return x, y_list, video_id_list

if __name__ == '__main__':
    data_path = sys.argv[1]
    un_X_train, un_Y_train, _ = load_data(os.path.join(data_path, 'unbal_train.h5'))
    ba_X_train, ba_Y_train, _ = load_data(os.path.join(data_path, 'bal_train.h5'))
    X_test, Y_test, _ = load_data(os.path.join(data_path, 'eval.h5'))
    
    X_train = np.concatenate((un_X_train, ba_X_train))
    del un_X_train, ba_X_train
    Y_train = un_Y_train + ba_Y_train
    pickle.dump((X_train, Y_train), open(os.path.join(data_path,'train_data.pkl'),'wb'),protocol=4)
    pickle.dump((X_test, Y_test), open(os.path.join(data_path,'test_data.pkl'),'wb'),protocol=4)
    

