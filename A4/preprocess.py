import os

import numpy as np
import pickle as pkl
import argparse

# Load Train and Dev Feats

INF = 9999999

def normalize_data(train_feats, dev_feats):
    # subtract mean, divide by std

    classes = sorted(list(train_feats.keys()))
    cl0_fns = sorted(list(train_feats[classes[0]].keys()))
    feat_ct = train_feats[classes[0]][cl0_fns[0]]
    all_data = np.empty((0,feat_ct))
    
    for cl in train_feats.keys():
        for fn in train_feats[cl].keys():
            all_data = np.vstack((all_data, train_feats[cl][fn]))
        # all_data = np.vstack((all_data,np.reshape(train_feats[cl], [-1,feat_ct])))
    print("all_data shape:", all_data.shape)
    mean_vec = np.mean(all_data, axis=0)
    std_vec = np.std(all_data, axis=0)
    
    for cl in train_feats.keys():
        for fn in train_feats[cl].keys():
            train_feats[cl][fn] = (train_feats[cl][fn] - mean_vec)/std_vec
            
    for cl in dev_feats.keys():
        for fn in dev_feats[cl].keys():
            dev_feats[cl][fn] = (dev_feats[cl][fn] - mean_vec)/std_vec

    return train_feats, dev_feats

def preprocess_synth():
    
    classes = [1,2]
    train_feats = {}
    dev_feats = {}

    train_data = np.loadtxt("./Data/Synthetic/train.txt",delimiter=",")
    for cl in classes:
        feats_here = train_data[np.where(train_data[:,2]==cl)][:,:2]
        train_feats[cl] = {i:feats_here[i] for i in range(feats_here.shape[0])}

    # train_feats = {cl:train_data[np.where(train_data[:,2]==cl)][:,:2] for cl in classes}
    dev_data = np.loadtxt("./Data/Synthetic/dev.txt",delimiter=",")
    for cl in classes:
        feats_here = dev_data[np.where(dev_data[:,2]==cl)][:,:2]
        dev_feats[cl] = {i:feats_here[i] for i in range(feats_here.shape[0])}

    # dev_feats = {cl:dev_data[np.where(dev_data[:,2]==cl)][:,:2] for cl in classes}
    return train_feats, dev_feats

def preprocess_image(norm=False):
    classes = ["coast","forest","highway","mountain","opencountry"]
    train_feats = {}
    dev_feats = {}

    for cl in classes:
        fns_train = sorted(os.listdir(f"./Data/ImageFeatures/{cl}/train/"))
        train_feats[cl] = {fn.split('.')[0]:np.loadtxt(f"./Data/ImageFeatures/{cl}/train/{fn}") for fn in fns_train}
        
        fns_dev = sorted(os.listdir(f"./Data/ImageFeatures/{cl}/dev/"))
        dev_feats[cl] = {fn.split('.')[0]:np.loadtxt(f"./Data/ImageFeatures/{cl}/dev/{fn}") for fn in fns_dev}

    if norm:
        train_feats, dev_feats = normalize_data(train_feats, dev_feats)

    return train_feats, dev_feats

def preprocess_digit():
    classes = [2,4,6,8,9]
    train_feats = {}
    dev_feats = {}

    for cl in classes:
        train_path = f"./Data/IsolatedDigits/{cl}/train/"
        dev_path = f"./Data/IsolatedDigits/{cl}/dev/"
        
        # Train feats
        train_fps = os.listdir(train_path)
        file_fps = [fp for fp in train_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
        train_feats[cl] = {}
        for fp in file_fps:
            fn = fp.split('.')[0]
            train_feats[cl][fn] = np.loadtxt(train_path+fp, skiprows=1)
        
        # Dev feats
        dev_fps = os.listdir(dev_path)
        file_fps = [fp for fp in dev_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
        dev_feats[cl] = {}
        for fp in file_fps:
            fn = fp.split('.')[0]
            dev_feats[cl][fn] = np.loadtxt(dev_path+fp, skiprows=1)

    return train_feats, dev_feats

def loadHW(path, angle=False):
    f = open(path, 'r')
    data = f.readline().split()[1:]
    data = [float(v) for v in data]
    feats = []
    for i in range(int(len(data)/2)):
        feats.append(np.array([data[2*i], data[2*i + 1]]))
    
    if angle:
        vecs = np.diff(np.array(feats), axis=0)
        angles = np.arctan2(vecs[:,1], vecs[:,0])
        return angles
    else:
        return feats

def preprocess_char(hwr_angle=True):
    classes = ['a', 'ai', 'bA', 'chA', 'dA']

    # Load Train and Dev

    train_feats = {}
    dev_feats = {}

    for cl in classes:
        train_path = f"./Data/HandwritingData/{cl}/train/"
        dev_path = f"./Data/HandwritingData/{cl}/dev/"
        
        # Train MFCCs
        train_fps = os.listdir(train_path)
        train_feats[cl] = {}
        for fp in train_fps:
            fn = fp.split('.')[0]

            feats_here = loadHW(train_path+fp, hwr_angle)
            feats_here = np.array(feats_here)
            if not hwr_angle:
                feats_here = feats_here - np.mean(feats_here, axis=0)
                feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

            train_feats[cl][fn] = feats_here
        
        # Dev MFCCs
        dev_fps = os.listdir(dev_path)
        dev_feats[cl] = {}
        for fp in dev_fps:
            fn = fp.split('.')[0]

            feats_here = loadHW(dev_path+fp, hwr_angle)
            feats_here = np.array(feats_here)
            if not hwr_angle:
                feats_here = feats_here - np.mean(feats_here, axis=0)
                feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

            dev_feats[cl][fn] = feats_here

    return train_feats, dev_feats

def pca(train_feats, dev_feats, dims=1):
    # TODO PCA
    return train_feats, dev_feats

def lda(train_feats, dev_feats, dims=1):
    # TODO LDA
    return train_feats, dev_feats

def raw(train_feats, dev_feats):
    return train_feats, dev_feats

def preprocess_choice(pr_type, algo):
    return eval(f'{algo}(*preprocess_{pr_type}())')

def save_feats(feats, fp):
    pkl.dump(feats, fp)

if __name__ == '__main__':
    algos = ['raw', 'pca', 'lda']
    pr_types = ['synth', 'image', 'digit', 'char']

    # pr_type = 'synth'
    # algo = 'raw'

    for pr in pr_types:
        for algo in algos:
            train_feats, dev_feats = preprocess_choice(pr, algo)
            with open(f'./Data/Pickles/{pr}_{algo}_train.pkl', 'wb') as f:
                save_feats(train_feats, f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev.pkl', 'wb') as f:
                save_feats(dev_feats, f)

            print(f"Preprocessed {algo} {pr}")