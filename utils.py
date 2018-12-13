#!/usr/bin/env python
# coding=utf-8
import h5py
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras 
import pdb 
import os 
import time 
import pickle

class CityData(tf.data.Dataset):
    pass

class CustomDataset(object):
    """
        Base class for generate dataset
    """
    def __init__(self, file_path,transform=None):
        """
            Args:
                file_path : the path to the training / validation set
                transform : a callable object that do the transform to the input data
        """
        assert os.path.exists(file_path), '{} no such file or directoy'\
                .format(file_path)
        print("loading dataset...")
        fid = h5py.File(file_path, 'r')
        self.s1 = fid['sen1']
        self.s2 = fid['sen2']
        # convert one hot coding from m x 17 to m
        self.labels = np.array(fid['label']).argmax(axis=1)
        print("finishing loading data")
        self.transform = transform
        self.drop_last = False
    
    def get_items(self, start_pos, end_pos):
        item = np.concatenate([self.s1[start_pos:end_pos,:], 
                               self.s2[start_pos:end_pos,:]], axis=3)
        if self.transform:
            item = self.transform(item)
        return item, np.array(self.labels)[start_pos:end_pos]    

    def load_data(self, batch_size=1, shuffle=False, drop_last=False):
        num_samples = self.labels.shape[0]
        self.drop_last = drop_last
        if shuffle:
            order = np.random.permutation(num_samples).tolist()
            batch = []
            labels = []
            for idx in order:
                item, label = self.get_items(idx,idx+1)
                batch.append(item)
                labels.append(label)
                if len(batch) == batch_size:
                    yield np.concatenate(batch, axis = 0), \
                            np.concatenate(labels, axis=0)
                    batch , labels = [] , []
            if len(batch) > 0 and not self.drop_last:
                yield np.concatenate(batch, axis=0), np.array(labels)
                    
        else:
            for current_pos in range(0, num_samples,batch_size):
                samples, labels = self.get_items(current_pos, current_pos+batch_size)
                if self.transform:
                    samples = self.transform(samples)
                yield samples, labels
            if not drop_last:
                samples , labels = self.get_items(current_pos, num_samples)
                yield samples, labels
    
    
if __name__ == '__main__':
    # just for debug the dataset 
    tic = time.time()
    background = np.zeros((32,32,18))
    datasets = CustomDataset('training.h5')
    counts = 0
    for samples, labels in datasets.load_data(batch_size=256,
                                              shuffle=False):
        background += np.sum(samples, axis = 0)
        counts += labels.shape[0]
        print("counts:",counts)
    mean = np.mean(background, axis=2)
    print("mean:", mean.shape)
    toc = time.time()
    print("elasped time is %.3f"%(toc - tic))
    with open("mean.pkl", 'wb') as f:
        pickle.dump(mean,f)
    

