# %%
# Assignment 3 Part B -- DTW and HMM

# %%
# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import scipy.stats as spst
import sklearn.metrics

from collections import Counter

# %%
# Load Train and Dev MFCCs
# Structure of train_mfcc and dev_mfcc = {digit:{filename:np_array_of_MFCC}}
INF = 9999999
digits = [2,4,6,8,9]
train_mfcc = {}
dev_mfcc = {}
pr_type = 'mfcc'
algo = 'dtw'

for dig in digits:
    train_path = f"./IsolatedDigits/{dig}/train/"
    dev_path = f"./IsolatedDigits/{dig}/dev/"
    
    # Train MFCCs
    train_fps = os.listdir(train_path)
    mfcc_fps = [fp for fp in train_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
    train_mfcc[dig] = {}
    for fp in mfcc_fps:
        fn = fp.split('.')[0]
        train_mfcc[dig][fn] = np.loadtxt(train_path+fp, skiprows=1)
    
    # Dev MFCCs
    dev_fps = os.listdir(dev_path)
    mfcc_fps = [fp for fp in dev_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
    dev_mfcc[dig] = {}
    for fp in mfcc_fps:
        fn = fp.split('.')[0]
        dev_mfcc[dig][fn] = np.loadtxt(dev_path+fp, skiprows=1)

# %%
def cost(v1,v2,angle=False):
    '''
    Computes cost of difference between two vectors
    '''

    if not angle:
        return np.linalg.norm(v1-v2)
    else:
        diff = v2-v1
        # diff[np.where(diff>np.pi)] -= 2*np.pi
        # diff[np.where(diff<-np.pi)] += 2*np.pi
        if diff > np.pi:
            diff -= 2*np.pi 
        if diff < -np.pi:
            diff += 2*np.pi
        return np.abs(diff)

def dtw(x, y, angle=False):
    '''
    Computes DTW between sequences of MFCC vectors x and y 
    '''
    
    XLEN = len(x)
    YLEN = len(y)
    
    dp = [[INF for j in range(YLEN+1)] for i in range(XLEN+1)]
    dp[0][0] = 0

    for i in range(1,XLEN+1):
        for j in range(1,YLEN+1):
            cost_here = cost(x[i-1], y[j-1], angle)
            dp[i][j] = cost_here + min([dp[i-1][j], dp[i][j-1], dp[i-1][j-1]])
    
    return dp[-1][-1]

def majorityVoting(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def dtwWithPath(x,y, angle=False):
    '''
    Same as dtw() but also return the warped form of y wrt x
    '''

    XLEN = len(x)
    YLEN = len(y)
    
    dp = [[[INF,i,j] for j in range(YLEN+1)] for i in range(XLEN+1)]
    dp[0][0] = [0,None,None]

    # Find DTW Matrix
    for i in range(1, XLEN+1):
        for j in range(1, YLEN+1):
            c = cost(x[i-1], y[j-1], angle)
            dp[i][j] = min([ [dp[i-1][j][0]+c, i-1, j], [dp[i][j-1][0]+c, i, j-1], [dp[i-1][j-1][0]+c, i-1, j-1]  ])

    # Extract Path
    curr = [XLEN, YLEN]
    path = [curr.copy()]
    dirs = []
    next = dp[curr[0]][curr[1]][1:]

    # ct = 0
    while next[0] != None and next[1] != None: 
        if next[0] < curr[0] and next[1] < curr[1]:
            dirs.append('J')
        elif next[1] < curr[1]:
            dirs.append('R')
        else:
            dirs.append('D')

        path.append(next.copy())
        curr = next.copy()
        next = dp[curr[0]][curr[1]][1:]

    path = path[::-1][1:]
    dirs = dirs[::-1][1:]
    
    # R : avg from curr to curr+runlength(R) inclusive
    # D : interp between curr and curr+runlength(D) inclusive, runlength(D) times

    # CWRT
    # Resolve all Rs first, they are avgs and their endpoints dont change
    # Then resolve Ds using the updated endpoints. Helps in case of ...DR...
    
    # Resolving Rs
    new_dirs = []
    warped_y = []
    R_run = 0
    D_run = 0
    ptr = 0
    for dir_ind in range(len(dirs)):
        dir = dirs[dir_ind]
        if dir == 'J':
            new_dirs.append('J')
            if R_run != 0:
                warped_y.append(np.mean(y[ptr-R_run:ptr+1], axis=0))
                R_run = 0
            elif D_run != 0:
                warped_y.append(y[ptr])
                D_run = 0
            else:
                warped_y.append(y[ptr])
            ptr+=1
        elif dir == 'R':
            R_run += 1
            ptr += 1
        else: # dir == 'D'
            D_run += 1
            new_dirs.append('D')
    if D_run != 0:
        warped_y.append(y[ptr])
        
    elif R_run != 0:
        warped_y.append(np.mean(y[ptr-R_run:ptr+1], axis=0))
    
    warped_y.append(y[-1])
    dirs = new_dirs.copy()

    # Resolving Ds
    y = warped_y.copy()
    warped_y = []
    D_run = 0
    ptr = 0
    for dir_ind in range(len(dirs)):
        dir = dirs[dir_ind]
        if dir == 'J':
            if D_run != 0:
                interps = np.linspace(y[ptr], y[ptr+1], D_run+2)[:-1]
                interps = [interps[ind] for ind in range(interps.shape[0])]
                warped_y.extend(interps)
                D_run = 0
                ptr+=1
            else:
                warped_y.append(y[ptr])
                ptr+=1
        else: # dir == 'D'
            D_run += 1
    if D_run != 0:
        interps = np.linspace(y[ptr], y[ptr+1], D_run+2)[:-1]
        interps = [interps[ind] for ind in range(interps.shape[0])]
        warped_y.extend(interps)
        D_run = 0
    else:
        warped_y.append(y[-1])
    
    return dp[-1][-1], warped_y


# %%
# test_arr1 = np.array([1,7,3,4,1,10,5,4,7,4])
# test_arr2 = np.array([1,4,5,10,9,3,2,6,8,4])
# print(cost(test_arr1[1],test_arr2[1]))
# res = dtwWithPath(test_arr1, test_arr2)
# print(f"warped y: {res[1]}")

# %%
def getTemplates(train_mfcc, angle=False):
    '''
    Computes a reference template for each class using CWRT
    '''

    templ = {dig:None for dig in train_mfcc.keys()}
    for dig in train_mfcc.keys():
        
        dig_fns = sorted(train_mfcc[dig].keys())
        init_templ = train_mfcc[dig][dig_fns[0]]
        warped_collection = [init_templ]
        for fn in dig_fns[1:]:
            mfcc_here = train_mfcc[dig][fn]
            res = dtwWithPath(init_templ, mfcc_here, angle)
            warped_mfcc_here = res[1]
            
            warped_collection.append(warped_mfcc_here)
            
        # print('shapes:', [len(arr) for arr in warped_collection])
        new_templ = np.mean(np.array(warped_collection), axis=0)
        templ[dig] = new_templ
    
    return templ
            

# %%
# Warp kc of Nc train examples of class c to same audio, then average
templ = getTemplates(train_mfcc)
print("Templates Built")

# %%
# Compute DTW error, get majority votes of top K, get class

def plotConfMat(conf_mat, sorted_cls, pr_type='', algo='', extra='', sf=2):
    # Plot confusion matrix
    sf = 2
    fig = plt.figure(figsize=(2.5*sf,2.5*sf))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3, 
                    interpolation='nearest')

    width, height = conf_mat.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(int(conf_mat[x][y])), xy=(y, x), ha='center', va='center', size=6*sf)

    cb = fig.colorbar(res)
    cb.ax.tick_params(labelsize=6*sf)
    plt.xticks(range(width), sorted_cls, fontsize=6*sf)
    plt.yticks(range(height), sorted_cls, fontsize=6*sf)
    plt.xlabel('Predicted', fontsize=6*sf)
    plt.ylabel('True', fontsize=6*sf)
    plt.title(f"Confusion Matrix {pr_type} {algo} {extra}", fontsize=6*sf)
    plt.savefig(f'./Plots/conf_{pr_type}_{algo}_{extra}.png', format='png')
    plt.show()

def predictCWRT(train_mfcc, dev_mfcc, angle=False, pr_type=''):
    algo = 'cwrt_dtw'
    t1 = time.time()
    correct = 0
    total = 0
    posts_list = []
    sorted_cls = sorted(list(train_mfcc.keys()))
    cl_ct = len(sorted_cls)
    conf_mat = np.zeros([cl_ct,cl_ct])
    c2i = {cl:sorted_cls.index(cl) for cl in sorted_cls}

    for dev_dig in dev_mfcc.keys():
        for dev_fn in dev_mfcc[dev_dig].keys():
            dev_frames = dev_mfcc[dev_dig][dev_fn]
            print(f"Dev {dev_fn} prediction: ", end = '')
            
            dig_best = None
            err_best = INF
            for train_dig in train_mfcc.keys():
                template = templ[train_dig]
                err_here = dtw(template, dev_frames, angle)
                posts_list.append([-err_here, train_dig, dev_dig])
                if err_here < err_best:
                    err_best = err_here
                    dig_best = train_dig
            conf_mat[c2i[dev_dig], c2i[dig_best]] += 1
            print(dig_best)
            if dig_best == dev_dig:
                correct+=1
            else:
                print(f"Failed, true: {dev_dig}, pred: {dig_best}")
            
            total += 1
    
    t2 = time.time()
    acc = correct/total
    print(f"Acc: {acc}")
    print("Time taken: {:.4f} sec".format(t2-t1))

    plotConfMat(conf_mat, sorted_cls, pr_type, algo, extra=f'angle_{angle}')

    return posts_list

def predict(train_mfcc, dev_mfcc, angle=False, pr_type=''):
    algo = 'direct_dtw'
    t1 = time.time()
    K = 5
    correct = 0
    total = 0
    
    posts_list = []
    sorted_cls = sorted(list(train_mfcc.keys()))
    cl_ct = len(sorted_cls)
    conf_mat = np.zeros([cl_ct,cl_ct])
    c2i = {cl:sorted_cls.index(cl) for cl in sorted_cls}

    for dev_dig in dev_mfcc.keys():
        for dev_fn in dev_mfcc[dev_dig].keys():
            dev_frames = dev_mfcc[dev_dig][dev_fn]
            print(f"Dev {dev_fn} prediction: ", end = '')
            
            dig_best = None
            err_best = INF
            errs = []
            for train_dig in train_mfcc.keys():
                best_train_err = INF
                for train_fn in train_mfcc[train_dig].keys():
                    
                    train_frames = train_mfcc[train_dig][train_fn]
                    err_here = dtw(train_frames, dev_frames, angle)
                    if err_here < best_train_err:
                        best_train_err = err_here
                    errs.append([err_here, train_dig])
                posts_list.append([-best_train_err, train_dig, dev_dig])
            topK = sorted(errs)[:K]
            topK_digs = [val[1] for val in topK]
            dig_best = majorityVoting(topK_digs)
            
            conf_mat[c2i[dev_dig], c2i[dig_best]] += 1
            print(dig_best)
            if dig_best == dev_dig:
                correct+=1
                print("Passed, votes:", topK_digs)
            else:
                print("Failed, votes:", topK_digs)
            total += 1
    
    t2 = time.time()
    acc = correct/total
    print(f"Acc: {acc}")
    print("Time taken: {:.4f} sec".format(t2-t1))

    # Plot confusion matrix
    plotConfMat(conf_mat, sorted_cls, pr_type, algo, extra=f'angle_{angle}')

    return posts_list


# %%
posts_list_CWRT = predictCWRT(train_mfcc, dev_mfcc, angle=False, pr_type=pr_type)
posts_ll = {'cwrt_dtw':posts_list_CWRT}
# posts_list_direct = predict(train_mfcc, dev_mfcc, angle=False, pr_type=pr_type)
# posts_ll = {'cwrt_dtw':posts_list_CWRT, 'direct_dtw':posts_list_direct}

# %%
# DTW Plots ROC, DET

def roc_det(posts_ll, cases, pr_type='', algo='', extra='', skip=1):

    # Plot ROC for select k
    fig1, ax1 = plt.subplots(figsize=[10,10])
    fig2, ax2 = plt.subplots(figsize=[10,10])

    # Sort in ascending order
    for c_id in range(len(cases)):
        posts_list = posts_ll[cases[c_id]]
        posts_list = sorted(posts_list)
        tp_fn_cts = [0]  #  tp_fn_cts[i] = no. of examples left of (and excluding) i which can be only tp or fn
        tn_fp_cts = [0]

        for i in range(len(posts_list)):
            post_here = posts_list[i]
            if post_here[1] == post_here[2]:
                tp_fn_cts.append(tp_fn_cts[-1] + 1)
                tn_fp_cts.append(tn_fp_cts[-1])
            else:
                tp_fn_cts.append(tp_fn_cts[-1])
                tn_fp_cts.append(tn_fp_cts[-1] + 1)

        # print(tp_fn_cts)
        # print(tn_fp_cts)
        tp_fn_tot = tp_fn_cts[-1]
        tn_fp_tot = tn_fp_cts[-1]
        tpr = []
        fpr = []
        tnr = []
        fnr = []
        
        for i in range(len(posts_list)):

            thresh = posts_list[i][0]
            tpr.append((tp_fn_tot - tp_fn_cts[i])/tp_fn_tot)
            fpr.append((tn_fp_tot - tn_fp_cts[i])/tn_fp_tot)
            tnr.append(tn_fp_cts[i]/tn_fp_tot)
            fnr.append(tp_fn_cts[i]/tp_fn_tot)

        # roc_pts = [[fpr[i],tpr[i]] for i in range(len(tpr))]
        if c_id%skip == 0:
            ax1.plot(fpr, tpr)
            ax2.plot(fpr, fnr)
            # ax2.plot(spst.norm.ppf(fpr), spst.norm.ppf(fnr))
            display = sklearn.metrics.DetCurveDisplay(fpr=fpr, fnr=fnr)

    
    # ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    # tick_locations = spst.norm.ppf(ticks)
    # tick_labels = [
    #     "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
    #     for s in ticks
    # ]
    # ax2.set_xticks(tick_locations)
    # ax2.set_xticklabels(tick_labels)
    # ax2.set_xlim(-3, 3)
    # ax2.set_yticks(tick_locations)
    # ax2.set_yticklabels(tick_labels)
    # ax2.set_ylim(-3, 3)

    ax1.legend(cases[::skip])
    ax2.legend(cases[::skip])
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')

    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('False Negative Rate (FNR)')

    ax2.set_xscale('logit')
    ax2.set_yscale('logit')

    # scale = 2
    # ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    # ax2.xaxis.set_major_formatter(ticks)
    # ax2.yaxis.set_major_formatter(ticks)

    ax1.set_title(f"ROC Curves {pr_type} {algo} {extra}")
    ax2.set_title(f"DET Curves {pr_type} {algo} {extra}")

    fig1.savefig(f"./Plots/ROC_{pr_type}_{algo}_{extra}.png")
    fig2.savefig(f"./Plots/DET_{pr_type}_{algo}_{extra}.png")
    plt.show()

    plt.figure()
    display.plot()
    plt.show()

# %%
roc_det(posts_ll, ['cwrt_dtw'], pr_type, 'dtw', extra='sole_cwrt')
# roc_det(posts_ll, ['cwrt_dtw', 'direct_dtw'], pr_type, 'dtw')

# %%
# Online Handwritten Character Recognition

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

# %%
letters = ['a', 'ai', 'bA', 'chA', 'dA']
pr_type='hwr'
hwr_angle = True

# Load Train and Dev

train_feats = {}
dev_feats = {}

for let in letters:
    train_path = f"./HandwritingData/{let}/train/"
    dev_path = f"./HandwritingData/{let}/dev/"
    
    # Train MFCCs
    train_fps = os.listdir(train_path)
    train_feats[let] = {}
    for fp in train_fps:
        fn = fp.split('.')[0]

        feats_here = loadHW(train_path+fp, hwr_angle)
        feats_here = np.array(feats_here)
        if not hwr_angle:
            feats_here = feats_here - np.mean(feats_here, axis=0)
            feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

        train_feats[let][fn] = feats_here
    
    # Dev MFCCs
    dev_fps = os.listdir(dev_path)
    dev_feats[let] = {}
    for fp in dev_fps:
        fn = fp.split('.')[0]

        feats_here = loadHW(dev_path+fp, hwr_angle)
        feats_here = np.array(feats_here)
        if not hwr_angle:
            feats_here = feats_here - np.mean(feats_here, axis=0)
            feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

        dev_feats[let][fn] = feats_here

# %%
# Just exploring the data
if not hwr_angle:
    eg_ct = 1
    for let in train_feats.keys():
        for fn in list(train_feats[let].keys())[:eg_ct]:
            pts = train_feats[let][fn]
            x_pts = [v[0] for v in pts]
            y_pts = [v[1] for v in pts]

            plt.figure()
            plt.scatter(x_pts,y_pts)
            plt.title(f"Plot of character {let} from file {fn}")
            plt.savefig(f"./Plots/chr_plot_{let}_{fn}.png")
            plt.show()

# %%
templ = getTemplates(train_feats, angle=hwr_angle)
print("Templates Built")

# %%
posts_list_CWRT = predictCWRT(train_feats, dev_feats, angle=hwr_angle, pr_type=pr_type)
posts_ll = {'cwrt_dtw':posts_list_CWRT}
# posts_list_direct = predict(train_feats, dev_feats, angle=hwr_angle, pr_type=pr_type)
# posts_ll = {'direct_dtw':posts_list_direct, 'cwrt_dtw':posts_list_CWRT}

# %%
# HWR DTW Plots
roc_det(posts_ll, ['cwrt_dtw'], pr_type, 'dtw', extra=f'angle_{hwr_angle}_sole_cwrt')
# roc_det(posts_ll, ['cwrt_dtw', 'direct_dtw'], pr_type, 'dtw', extra=f'angle_{hwr_angle}')

# %%
# Discrete HMM

# 1. Use vector quantization (k-means) to convert feature vecs to symbols
# 2. Store train and test symbol seqs in files
# 3. Use HMM C++ code to train on train seqs, build model
# 4. Use HMM C++ code to eval trained model on test seqs

# %%
# Vector Quantization using k-means

def k_means(data,k=15):
    # reshape data for numpy broadcasting
    data = np.reshape(data,[-1,1,data.shape[-1]])
    # print(f"Data shape {data.shape}")
    # initialize k means with random means
    myrange = np.linspace(0,data.shape[0]-1,k,dtype = np.int64)
    kmeans = np.vstack([data[myrange,0,i] for i in range(0,data.shape[-1])]).T
    
    # reshape data for numpy broadcasting
    kmeans = np.reshape(kmeans,[1,-1,kmeans.shape[-1]])
    # print(f"kmeans shape {kmeans.shape}")
    # print(f"kmeans: {kmeans}")
    # calculate distances
    dists = np.sum((data - kmeans)**2,axis=2)
    # print(f"dists shape {dists.shape}")
    # initialize cluster assignments
    
    # choose n rows from a kxk identity matrix 
    # using an nx1 argmin matrix (ranging from 0 to k-1)
    # to produce an nxk 1-hot encoded matrix
    r_nk_old = np.eye(k)[np.argmin(dists,axis=1)]
    r_nk_new = r_nk_old.copy()
    # print(f"r_nk axis=0 sum: {np.sum(r_nk_new, axis=0)}")
    c = 0
    while True:
        #print(f"Iteration {c}")
        c+=1
        # move cluster assignments into old variable for comparison
        r_nk_old = r_nk_new.copy()
        # update means
        if np.any(np.sum(r_nk_old,axis=0) == 0):
            print(r_nk_old)
            print(f"r_nk axis=0 sum: {np.sum(r_nk_old, axis=0)}")
            print("error, 0 sum encountered")
            break

        kmeans = (r_nk_old.T @ np.reshape(data, [data.shape[0],data.shape[2]]))/np.reshape(np.sum(r_nk_old,axis=0),[-1,1])
        # update new cluster assignments
        dists = np.sum((data - kmeans)**2,axis=2)
        r_nk_new = np.eye(k)[np.argmin(dists,axis=1)]
        # test for convergence
        if np.all(r_nk_old == r_nk_new):
            break
    print(f"Iterations to convergence = {c}")
    # print(f"r_nk shape: {r_nk_new.shape}")
    r_nk_int = np.argmax(r_nk_new, axis=1)
    return kmeans, r_nk_new, r_nk_int

def quantize_with_means(feats, means):
    '''
    Quantizes each example, given k-means centres
    '''

    labels = {}
    for cl in feats.keys():
        labels[cl] = {}
        for fn in sorted(list(feats[cl].keys())):
            labels[cl][fn] = []
            feats_here = feats[cl][fn]
            if len(feats_here.shape) == 1:
                feats_here = np.reshape(feats_here, [-1,1])
            
            for feat in feats_here:
                min_norm = INF
                for m_id in range(len(means)):
                    # print("feat:", feat)
                    # print("means:", means)
                    norm_here = np.linalg.norm(feat-means[m_id])
                    if norm_here < min_norm:
                        min_norm = norm_here
                        feat_sym = m_id
                labels[cl][fn].append(feat_sym)
    return labels

def quantize_train(train_feats, k=15):
    '''
    Quantizes real valued vectors into symbols derived from k-means cluster centres
    '''

    # Take all vectors across files for particular class. Do k-means on that
    # For eg in handwriting, this means -- take one letter, do k-means on all instances of that letter
    
    means = None
    syms = {}
    all_feats = []
    tot = 0
    bd = {}
    cls = sorted(list(train_feats.keys()))
    for cl in cls:
        fns = sorted(list(train_feats[cl].keys()))
        if len(train_feats[cl][fns[0]].shape) == 1:
            class_feats = np.concatenate([np.reshape(train_feats[cl][fn], [-1,1]) for fn in fns])
            # print(class_feats.shape)
        else:
            class_feats = np.concatenate([train_feats[cl][fn] for fn in fns])
            # print(class_feats.shape)
        bounds = np.array([len(train_feats[cl][fn]) for fn in fns])
        bounds = [0] + list(np.cumsum(bounds))
        bd[cl] = {fns[i]:[bounds[i]+tot,bounds[i+1]+tot] for i in range(len(fns))}
        all_feats.append(class_feats)
        tot += len(class_feats)
    
    all_feats = np.concatenate(all_feats)
    # means, syms = k_means_sklearn(all_feats)
    means, syms_onehot, syms = k_means(all_feats, k)

    symd = {}
    for cl in train_feats.keys():
        symd[cl] = {}
        fns = sorted(list(train_feats[cl].keys()))
        for fn in fns:
            symd[cl][fn] = syms[bd[cl][fn][0]:bd[cl][fn][1]]
    
    return means, symd

def write_syms(symd, pref='dummy'):
    '''
    Write quantized symbols into seq file for HMM C++
    Set pref to 'train' or 'dev'
    '''

    try:
        os.mkdir(f'./HMM-Code/{pref}')
    except FileExistsError:
        pass
    
    for cl in symd.keys():
        out_str = ""
        for fn in symd[cl].keys():
            out_str += " ".join(str(char) for char in symd[cl][fn])
            out_str += "\n"

        with open(f'./HMM-Code/{pref}/{cl}.txt', 'w') as fo:
            fo.write(out_str)

# %%
letters = ['a', 'ai', 'bA', 'chA', 'dA']
hwr_angle = True

# Load Train and Dev Letters
# Structure of train_feats and dev_feats = {digit:{filename:np_array_of_feats}}

train_feats = {}
dev_feats = {}

for let in letters:
    train_path = f"./HandwritingData/{let}/train/"
    dev_path = f"./HandwritingData/{let}/dev/"
    
    # Train MFCCs
    train_fps = os.listdir(train_path)
    train_feats[let] = {}
    for fp in train_fps:
        fn = fp.split('.')[0]

        feats_here = loadHW(train_path+fp, hwr_angle)
        feats_here = np.array(feats_here)
        if not hwr_angle:
            feats_here = feats_here - np.mean(feats_here, axis=0)
            feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

        train_feats[let][fn] = feats_here
    
    # Dev MFCCs
    dev_fps = os.listdir(dev_path)
    dev_feats[let] = {}
    for fp in dev_fps:
        fn = fp.split('.')[0]

        feats_here = loadHW(dev_path+fp, hwr_angle)
        feats_here = np.array(feats_here)
        if not hwr_angle:
            feats_here = feats_here - np.mean(feats_here, axis=0)
            feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

        dev_feats[let][fn] = feats_here

# %%
# mean_debug, sym_debug = quantize_train(train_feats, 5)

# %%
# HWR HMM
# K = 10
acc_vs_k = []
pr_type = 'hwr'
algo = 'hmm'
k_range = list(range(5,25))
str_k_range = [str(v) for v in k_range]
posts_ll = {}
sorted_cls = sorted(list(train_feats.keys()))
cl_to_ind = {cl:sorted_cls.index(cl) for cl in sorted_cls}
for K in str_k_range:
    print(f"Starting K={K}")

    # List of copy examples for ROC calc later
    posts_ll[K] = []
    posts_list_k = []

    cl_ct = len(list(train_feats.keys()))

    if hwr_angle:
        sym_ct = 5
    else:
        sym_ct = K
    # Quantized
    means_train, syms_train = quantize_train(train_feats,sym_ct)
    # print(means_train)
    syms_dev = quantize_with_means(dev_feats, means_train)

    # Write symbols to sequences file for HMM C++
    write_syms(syms_train, f'{pr_type}_train')
    write_syms(syms_dev, f'{pr_type}_dev')

    # Set initial params for each class-specific HMM
    # Assume left-to-right HMM, and each state modelled by unimodal Gaussian
    # So num_states = num_symbols = k in k-means used earlier to cluster vectors 

    # Call HMM C++ trainer, get trained models for each class
    for cl in sorted(list(train_feats.keys())):

        sp = subprocess.run(["wsl", "cd", "HMM-Code;", "./train_hmm", f"./{pr_type}_train/{cl}.txt", "1234", f"{K}", f"{sym_ct}", "0.01"], shell=True, capture_output=True)
        # print(sp.stdout.decode('utf-8'))
        # print("Errors:", sp.stderr.decode('utf-8'))

    # Test on dev seqs, get probabs from each HMM, predict with argmax
    preds = {}
    for dev_cl in sorted(list(dev_feats.keys())):
        dev_cl_fns = sorted(list(dev_feats[dev_cl].keys()))
        dev_cl_fns_ind = {dev_fn:dev_cl_fns.index(dev_fn) for dev_fn in dev_cl_fns}
        preds[dev_cl] = {dev_ex:{} for dev_ex in dev_cl_fns}
        for train_cl in sorted(list(train_feats.keys())):
            sp = subprocess.run(["wsl", "cd", "HMM-Code;", "./test_hmm", f"./{pr_type}_dev/{dev_cl}.txt", f"./{pr_type}_train/{train_cl}.txt.hmm"], shell=True, capture_output=True)
            # print(sp.stdout.decode('utf-8'))
            out = sp.stdout.decode('utf-8')
            out_lst = out.split('\n')
            out_lst = [line for line in out_lst if line[:5]=='alpha']
            out_lst = [float(line.split(' ')[4]) for line in out_lst]
            # print(out_lst)
            
            for dev_ex in dev_cl_fns:
                preds[dev_cl][dev_ex][train_cl] = out_lst[dev_cl_fns_ind[dev_ex]]
                posts_list_k.append([out_lst[dev_cl_fns_ind[dev_ex]],train_cl,dev_cl])
            # print("Errors (if any):", sp.stderr.decode('utf-8'))
            # print("-----------------------------------------------")
    print("Probabs Generated")

    correct = 0
    total = 0
    conf_mat = np.zeros([cl_ct, cl_ct])
    preds_keys = sorted(list(preds.keys()))
    for i in range(cl_ct):
        dev_cl = preds_keys[i]
        preds_cl_keys = sorted(list(preds[dev_cl].keys()))
        for j in range(len(preds_cl_keys)):
            dev_fn = preds_cl_keys[j]
            probs_here = preds[dev_cl][dev_fn]
            pred_cl = max(probs_here, key=probs_here.get)
            conf_mat[i,cl_to_ind[pred_cl]] += 1
            if pred_cl == dev_cl:
                correct += 1
            total += 1
    acc = correct/total
    print(f"Acc: {acc}")
    acc_vs_k.append(acc)

    posts_ll[K] = posts_list_k.copy()
    
    # Plot Confusion Matrix
    plotConfMat(conf_mat, sorted_cls, pr_type, algo, extra=f'K_{K}_angle_{hwr_angle}')

# %%
# HWR HMM Plots (Acc vs k, ROC, DET)
plt.figure(figsize=[10,10])
plt.plot(k_range, acc_vs_k)
plt.title(f"Accuracy vs. k {pr_type} {algo} angle_{hwr_angle}")
plt.ylim(0.5,1.0)
plt.savefig(f"./Plots/Acc_vs_k_{pr_type}_{algo}_angle_{hwr_angle}.png")
plt.show()

# Plot ROC for select k
roc_det(posts_ll, str_k_range, 'hwr', 'hmm', extra=f'angle_{hwr_angle}', skip=2)

# %%
# Load Train and Dev MFCCs
# Structure of train_mfcc and dev_mfcc = {digit:{filename:np_array_of_MFCC}}
INF = 9999999
digits = [2,4,6,8,9]
train_mfcc = {}
dev_mfcc = {}

for dig in digits:
    train_path = f"./IsolatedDigits/{dig}/train/"
    dev_path = f"./IsolatedDigits/{dig}/dev/"
    
    # Train MFCCs
    train_fps = os.listdir(train_path)
    mfcc_fps = [fp for fp in train_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
    train_mfcc[dig] = {}
    for fp in mfcc_fps:
        fn = fp.split('.')[0]
        train_mfcc[dig][fn] = np.loadtxt(train_path+fp, skiprows=1)
    
    # Dev MFCCs
    dev_fps = os.listdir(dev_path)
    mfcc_fps = [fp for fp in dev_fps if fp[len(fp)-4:len(fp)] == 'mfcc']
    dev_mfcc[dig] = {}
    for fp in mfcc_fps:
        fn = fp.split('.')[0]
        dev_mfcc[dig][fn] = np.loadtxt(dev_path+fp, skiprows=1)

train_feats = train_mfcc
dev_feats = dev_mfcc

# %%
# MFCC HMM
# K = 10
acc_vs_k = []
pr_type = 'mfcc'
algo = 'hmm'
k_range = list(range(5,25))
str_k_range = [str(v) for v in k_range]
posts_ll = {}
sorted_cls = sorted(list(train_feats.keys()))
cl_to_ind = {cl:sorted_cls.index(cl) for cl in sorted_cls}
for K in str_k_range:
    print(f"Starting K={K}")

    # List of copy examples for ROC calc later
    posts_ll[K] = []
    posts_list_k = []

    cl_ct = len(list(train_feats.keys()))

    # Quantized
    means_train, syms_train = quantize_train(train_feats, int(K))
    # print(means_train)
    syms_dev = quantize_with_means(dev_feats, means_train)

    # Write symbols to sequences file for HMM C++
    write_syms(syms_train, 'mfcc_train')
    write_syms(syms_dev, 'mfcc_dev')

    # Set initial params for each class-specific HMM
    # Assume left-to-right HMM, and each state modelled by unimodal Gaussian
    # So num_states = num_symbols = k in k-means used earlier to cluster vectors 

    # Call HMM C++ trainer, get trained models for each class
    for cl in sorted(list(train_feats.keys())):

        sp = subprocess.run(["wsl", "cd", "HMM-Code;", "./train_hmm", f"./{pr_type}_train/{cl}.txt", "1234", f"{K}", f"{K}", "0.01"], shell=True, capture_output=True)
        # print(sp.stdout.decode('utf-8'))
        # print("Errors:", sp.stderr.decode('utf-8'))

    # Test on dev seqs, get probabs from each HMM, predict with argmax
    preds = {}
    for dev_cl in sorted(list(dev_feats.keys())):
        dev_cl_fns = sorted(list(dev_feats[dev_cl].keys()))
        dev_cl_fns_ind = {dev_fn:dev_cl_fns.index(dev_fn) for dev_fn in dev_cl_fns}
        preds[dev_cl] = {dev_ex:{} for dev_ex in dev_cl_fns}
        for train_cl in sorted(list(train_feats.keys())):
            sp = subprocess.run(["wsl", "cd", "HMM-Code;", "./test_hmm", f"./{pr_type}_dev/{dev_cl}.txt", f"./{pr_type}_train/{train_cl}.txt.hmm"], shell=True, capture_output=True)
            # print(sp.stdout.decode('utf-8'))
            out = sp.stdout.decode('utf-8')
            out_lst = out.split('\n')
            out_lst = [line for line in out_lst if line[:5]=='alpha']
            out_lst = [float(line.split(' ')[4]) for line in out_lst]
            # print(out_lst)
            
            for dev_ex in dev_cl_fns:
                preds[dev_cl][dev_ex][train_cl] = out_lst[dev_cl_fns_ind[dev_ex]]
                posts_list_k.append([out_lst[dev_cl_fns_ind[dev_ex]],train_cl,dev_cl])
            # print("Errors (if any):", sp.stderr.decode('utf-8'))
            # print("-----------------------------------------------")
    print("Probabs Generated")

    correct = 0
    total = 0
    conf_mat = np.zeros([cl_ct, cl_ct])
    preds_keys = sorted(list(preds.keys()))
    for i in range(cl_ct):
        dev_cl = preds_keys[i]
        preds_cl_keys = sorted(list(preds[dev_cl].keys()))
        for j in range(len(preds_cl_keys)):
            dev_fn = preds_cl_keys[j]
            probs_here = preds[dev_cl][dev_fn]
            pred_cl = max(probs_here, key=probs_here.get)
            conf_mat[i,cl_to_ind[pred_cl]] += 1
            if pred_cl == dev_cl:
                correct += 1
            total += 1
    acc = correct/total
    print(f"Acc: {acc}")
    acc_vs_k.append(acc)

    posts_ll[K] = posts_list_k.copy()

    # Plot Confusion Matrix
    plotConfMat(conf_mat, sorted_cls, pr_type, algo, extra=f'K_{K}')

# %%
# MFCC HMM Plots (Acc vs k, ROC, DET)
plt.figure(figsize=[10,10])
plt.plot(k_range, acc_vs_k)
plt.title(f"Accuracy vs. k {pr_type} {algo}")
plt.ylim(0.5,1.0)
plt.savefig(f"./Plots/Acc_vs_k_{pr_type}_{algo}.png")
plt.show()

# Plot ROC
roc_det(posts_ll, str_k_range, 'mfcc', 'hmm', skip=2)

# %%
# Connected Digits conn_mfcc (Unfinished)

# 1. Transform dev data into correct features
# 2. Concatenate HMMs 

# %%
letters = ['a', 'ai', 'bA', 'chA', 'dA']
hwr_angle = True
pr_type = 'conn_hwr'

# Load test and Dev Letters
# Structure of test_feats and dev_feats = {digit:{filename:np_array_of_feats}}

test_feats = {}
dev_feats = {}

test_path = f"./ConnectedHandwriting/test/"
dev_path = f"./ConnectedHandwriting/dev/"

# Test MFCCs
test_fps = os.listdir(test_path)
for fp in test_fps:
    fn = fp.split('.')[0]

    feats_here = loadHW(test_path+fp, hwr_angle)
    feats_here = np.array(feats_here)
    if not hwr_angle:
        feats_here = feats_here - np.mean(feats_here, axis=0)
        feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

    test_feats[fn] = feats_here

# Dev MFCCs
dev_fps = os.listdir(dev_path)
for fp in dev_fps:
    fn = fp.split('.')[0]

    feats_here = loadHW(dev_path+fp, hwr_angle)
    feats_here = np.array(feats_here)
    if not hwr_angle:
        feats_here = feats_here - np.mean(feats_here, axis=0)
        feats_here = feats_here/np.sqrt(np.var(feats_here, axis=0))

    dev_feats[fn] = feats_here

# %%
# Just exploring the data
if not hwr_angle:
    eg_ct = 5
    for fn in sorted(list(dev_feats.keys()))[:eg_ct]:
        
        pts = dev_feats[fn]
        x_pts = [v[0] for v in pts]
        y_pts = [v[1] for v in pts]

        plt.figure()
        plt.scatter(x_pts,y_pts)
        plt.title(f"Plot from file {fn}")
        plt.savefig(f"./Plots/conn_chr_dev_{fn}.png")
        plt.show()


