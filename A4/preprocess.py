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
    # print("all_data shape:", all_data.shape)
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

    # print(train_feats["coast"]["coast_arnat59"].shape)
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
            else:
                feats_here = np.reshape(feats_here, [-1,1])

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
            else:
                feats_here = np.reshape(feats_here, [-1,1])

            dev_feats[cl][fn] = feats_here

    return train_feats, dev_feats

def pca(train_feats, dev_feats, dims=100):
    classes = sorted(list(train_feats.keys()))
    cl0_fns = sorted(list(train_feats[classes[0]].keys()))
    try:
        feat_ct = train_feats[classes[0]][cl0_fns[0]].shape[1]
    except IndexError:
        feat_ct = train_feats[classes[0]][cl0_fns[0]].shape[0]

    all_data = np.empty((0,feat_ct))
    
    # Get all data in one array
    for cl in train_feats.keys():
        for fn in train_feats[cl].keys():
            # print("shape here:", train_feats[cl][fn].shape)
            all_data = np.vstack((all_data, train_feats[cl][fn]))
    
    mean_vec = np.mean(all_data, axis=0)
    centered_data = all_data - mean_vec
    
    # TODO cov vs. corr 
    # cov_mat = np.cov(centered_data, rowvar=False)
    cov_mat = np.corrcoef(centered_data, rowvar=False)
    # print("cov_mat shape:", cov_mat.shape)
    if len(cov_mat.shape) > 0 and cov_mat.shape[0] > 1:
        e_vals, e_vecs = np.linalg.eig(cov_mat)
        
        # Sort eigenvalues and eigenvectors by magnitude
        idx = np.abs(e_vals).argsort()[::-1]
        e_vals = e_vals[idx][:dims]
        e_vecs = e_vecs[:,idx][:,:dims]
        
        for cl in train_feats.keys():
            for fn in train_feats[cl].keys():
                train_feats[cl][fn] = (train_feats[cl][fn]-mean_vec) @ e_vecs
                
        for cl in dev_feats.keys():
            for fn in dev_feats[cl].keys():
                dev_feats[cl][fn] = (dev_feats[cl][fn]-mean_vec) @ e_vecs
        
    return train_feats,dev_feats

def lda(train_feats, dev_feats, dims=100):
    # TODO LDA
    classes = sorted(list(train_feats.keys()))
    cl0_fns = sorted(list(train_feats[classes[0]].keys()))
    try:
        feat_ct = train_feats[classes[0]][cl0_fns[0]].shape[1]
    except IndexError:
        feat_ct = train_feats[classes[0]][cl0_fns[0]].shape[0]

    all_data = np.empty((0,feat_ct))

    SW = np.zeros([feat_ct, feat_ct])
    SB = np.zeros([feat_ct, feat_ct])

    for cl in train_feats.keys():
        for fn in train_feats[cl].keys():
            all_data = np.vstack((all_data, train_feats[cl][fn]))
        # all_data = np.vstack((all_data,np.reshape(train_feats[cl], [-1,feat_ct])))
    print("all_data shape:", all_data.shape)
    mean_all = np.mean(all_data, axis=0)
    std_all = np.std(all_data, axis=0)
    mean_sep = None
    cl_means = {}
    for cl in classes:
        # find class mean
        cl_data = np.empty([0, feat_ct])
        fns = sorted(list(train_feats[cl].keys()))
        for fn in fns:
            cl_data = np.vstack((cl_data, train_feats[cl][fn]))
        cl_means[cl] = np.mean(cl_data, axis=0)

        SW += (cl_data-cl_means[cl]).T @ (cl_data-cl_means[cl])
        
        fn_ct = cl_data.shape[0]
        mean_sep = (cl_means[cl] - mean_all)
        SB += fn_ct * (mean_sep.T @ mean_sep)

    print("mean sep shape:", mean_sep.shape)
    print("Symm? : ", np.allclose(swi_sb, swi_sb.T, rtol=1e-4))
    print("Symm? : ", np.allclose(swi_sb, swi_sb.T, rtol=1e-4))
    swi_sb = np.linalg.inv(SW) @ SB
    print("Symm? : ", np.allclose(swi_sb, swi_sb.T, rtol=1e-4))
    e_vals, e_vecs = np.linalg.eig(swi_sb)
    print("e_vals dtype:", e_vals.dtype)
    print("e_vecs dtype:", e_vecs.dtype)

    idx = np.argsort(abs(e_vals))[::-1]
    e_vals = e_vals[idx][:dims]
    e_vecs = e_vecs[:,idx][:, :dims]

    print("e_vecs shape:", e_vecs.shape)
    for cl in sorted(list(train_feats.keys())):
        for fn in sorted(list(train_feats[cl].keys())):
            train_feats[cl][fn] = train_feats[cl][fn] @ e_vecs
            

    for cl in sorted(list(dev_feats.keys())):
        for fn in sorted(list(dev_feats[cl].keys())):
            dev_feats[cl][fn] = dev_feats[cl][fn] @ e_vecs

    return train_feats, dev_feats

def datadict2np(feats):
    classes = sorted(list(feats.keys()))
    cl2ind = {classes[i]:i for i in range(len(classes))}
    cl0_fns = sorted(list(feats[classes[0]].keys()))
    feat_shape = feats[classes[0]][cl0_fns[0]].shape
    if len(feat_shape) in [1,2]:
        feat_ct = np.prod(feat_shape)
    else:
        print("Feature shape unexpected, stopping")
        assert False

    X = np.empty([0,feat_ct])
    y = np.empty([0,1])
    if len(feat_shape) == 1:
        for cl in classes:
            cl_here = cl2ind[cl]
            fns = sorted(list(feats[cl].keys()))
            for fn in fns:
                feats_here = feats[cl][fn].reshape([1,-1])
                X = np.vstack((X, feats_here))
                y = np.vstack((y, cl_here))
    
    elif len(feat_shape) == 2:
        for cl in classes:
            cl_here = cl2ind[cl]
            fns = sorted(list(feats[cl].keys()))
            for fn in fns:
                feats_here = feats[cl][fn].reshape([1,-1])
                X = np.vstack((X, feats_here))
                y = np.vstack((y, cl_here))

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X,y

def raw(train_feats, dev_feats):
    return train_feats, dev_feats

def preprocess_choice(pr_type, algo):
    return eval(f'{algo}(*preprocess_{pr_type}())')

def save_feats(feats, fp):
    pkl.dump(feats, fp)

if __name__ == '__main__':
    algos = ['raw', 'pca', 'lda']
    # pr_types = ['synth', 'image', 'digit', 'char']
    pr_types = ['synth', 'image']

    for pr in pr_types:
        for algo in algos:
            train_feats, dev_feats = preprocess_choice(pr, algo)
            with open(f'./Data/Pickles/{pr}_{algo}_train.pkl', 'wb') as f:
                save_feats(train_feats, f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev.pkl', 'wb') as f:
                save_feats(dev_feats, f)

            X_train, y_train = datadict2np(train_feats)
            X_dev, y_dev = datadict2np(dev_feats)
            
            with open(f'./Data/Pickles/{pr}_{algo}_train_np_X.pkl', 'wb') as f:
                save_feats(X_train, f)

            with open(f'./Data/Pickles/{pr}_{algo}_train_np_y.pkl', 'wb') as f:
                save_feats(y_train, f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev_np_X.pkl', 'wb') as f:
                save_feats(X_dev, f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev_np_y.pkl', 'wb') as f:
                save_feats(y_dev, f)

            print(f"Preprocessed {algo} {pr}")