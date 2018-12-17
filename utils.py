#!/usr/bin/env python
# coding=utf-8
import h5py
import numpy as np
import tensorflow as tf 
import pdb 
import os 
import time 
import pickle
import transform as T

def unpickle(filename):
    """
        Args:
            unpickle the filenames
    """
    assert os.path.isfile(filename), "{} no such file or directoy"\
            .format(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data 

def pickle_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


class CityData(tf.data.Dataset):
    pass

class CustomDataset(object):
    """
        Base class for generate dataset
    """
    def __init__(self, file_path,transform=None):
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
    
    def __len__(self):
        return self.labels.shape[0]

    def get_items(self, start_pos, end_pos):
        item = np.concatenate([self.s1[start_pos:end_pos,:], 
                               self.s2[start_pos:end_pos,:]], axis=3)
        if self.transform:
            item = self.transform(item)
        return item, np.array(self.labels)[start_pos:end_pos]    

    def load_data(self, batch_size=1, shuffle=False, drop_last=False):
        num_samples = self.labels.shape[0]
        self.drop_last = drop_last
        while True:
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
                    if drop_last and current_pos >= num_samples - batch_size:
                        break
                    samples, labels = self.get_items(current_pos, current_pos+batch_size)
                    if self.transform:
                        samples = self.transform(samples)
                    yield samples, labels
    
    
if __name__ == '__main__':
    # just for debug the dataset 
    tic = time.time()
    background = np.ones(18,)
    background = -9999 * background
    counts = 0
    mean = unpickle('mean_channal.pkl')
    tfs = T.Compose([
        T.Normalize(mean=mean),
        T.RandomHorizontalFlip(0.5),
    ])
    datasets = CustomDataset('training.h5', transform=None)
    for samples, labels in datasets.load_data(batch_size=256,
                                              shuffle=False):
        # background += np.sum(samples, axis = (0,1,2))
        counts += labels.shape[0]
        print("counts:",counts)
    # compute mean,std by channel
    print("len:", len(datasets))
    # mean = background / len(datasets)
    # print("mean:", mean.shape)
    toc = time.time()
    print("elasped time is %.3f"%(toc - tic))
    # pdb.set_trace()
    # all_data = np.concatenate([datasets.s1,datasets.s2], axis=3)
    # with open("mean_channal.pkl", 'wb') as f:
    #    pickle.dump(mean,f)

    

