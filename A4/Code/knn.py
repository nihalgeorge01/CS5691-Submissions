# TODO K Nearest Neighbours
# 
# Load features from file, then apply model <<< DONE >>>
# TODO Try diff params, give accuracies, misclassification rates for each dataset
# TODO Plot confusion matrix, ROC, DET

# feature type - raw, pca, lda
# dataset - synth, image, digit, char

from threading import Thread

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.stats as spst

INF = 9999999999

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def get_posts_list_from_scores(scores, y_dev):
    posts_list = []

    for i in range(scores.shape[0]):
        posts_list.extend([[scores[i,j], j, y_dev[i]] for j in range(scores.shape[1])])
    
    return posts_list

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
            # ax2.plot(fpr, fnr)
            ax2.plot(spst.norm.ppf(fpr), spst.norm.ppf(fnr))
            # display = sklearn.metrics.DetCurveDisplay(fpr=fpr, fnr=fnr)

    
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = spst.norm.ppf(ticks)
    tick_labels = [
        "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
        for s in ticks
    ]
    ax2.set_xticks(tick_locations)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(-3, 3)
    ax2.set_yticks(tick_locations)
    ax2.set_yticklabels(tick_labels)
    ax2.set_ylim(-3, 3)

    ax1.legend(cases[::skip])
    ax2.legend(cases[::skip])
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')

    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('False Negative Rate (FNR)')

    # ax2.set_xscale('logit')
    # ax2.set_yscale('logit')

    # scale = 2
    # ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    # ax2.xaxis.set_major_formatter(ticks)
    # ax2.yaxis.set_major_formatter(ticks)

    ax1.set_title(f"ROC Curves {pr_type} {algo} {extra}")
    ax2.set_title(f"DET Curves {pr_type} {algo} {extra}")

    fig1.savefig(f"./Plots/ROC_{pr_type}_{algo}_{extra}.png")
    fig2.savefig(f"./Plots/DET_{pr_type}_{algo}_{extra}.png")
    plt.show()

def euclidean(x,y):
    return np.linalg.norm(x-y)

def cost(v1,v2,angle=False):
    '''
    Computes cost of difference between two vectors
    '''

    if not angle:
        return euclidean(v1,v2)
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

    w = 20

    for i in range(1,XLEN+1):
        for j in range(max(1, i-w), min(YLEN+1, i+w)):
            dp[i][j] = 0
    
    for i in range(1,XLEN+1):
        for j in range(max(1,i-w),min(YLEN+1, i+w)):
            cost_here = cost(x[i-1], y[j-1], angle)
            dp[i][j] = cost_here + min([dp[i-1][j], dp[i][j-1], dp[i-1][j-1]])
    
    return dp[-1][-1]

def dtwAngle(x, y, angle=True):
    '''
    Computes DTW between sequences of MFCC vectors x and y 
    '''
    
    XLEN = len(x)
    YLEN = len(y)
    
    dp = [[INF for j in range(YLEN+1)] for i in range(XLEN+1)]
    dp[0][0] = 0

    w = 100

    for i in range(1,XLEN+1):
        for j in range(max(1, i-w), min(YLEN+1, i+w)):
            dp[i][j] = 0
    
    for i in range(1,XLEN+1):
        for j in range(max(1,i-w),min(YLEN+1, i+w)):
            cost_here = cost(x[i-1], y[j-1], angle)
            dp[i][j] = cost_here + min([dp[i-1][j], dp[i][j-1], dp[i-1][j-1]])
    
    return dp[-1][-1]

def knn(train_feats, dev_feats, distfun=euclidean, k=5):
    # Tag each example with its class

    # Find k min distances between test and all train examples
    # Dist function varies -- For synth its direct euclidean
    # For image, find squared distance b/w corresponding blocks, add them up, then rank
    # For digit it's dtw
    # For char it's dtw

    correct = 0
    total = 0

    posts_list = []

    for dev_cl in sorted(list(dev_feats.keys())):
        for dev_fn in sorted(list(dev_feats[dev_cl].keys())):
            df = dev_feats[dev_cl][dev_fn]
            
            dev_dists = []
            for train_cl in sorted(list(train_feats.keys())):
                for train_fn in sorted(list(train_feats[train_cl].keys())):
                    tf = train_feats[train_cl][train_fn]
                    
                    dev_dists.append([distfun(tf,df), train_cl])
            
            topk = sorted(dev_dists)[:k]
            votes = {}
            for cand in topk:
                if cand[1] in votes:
                    votes[cand[1]] += 1
                else:
                    votes[cand[1]] = 1
            
            pred_cl = max(votes, key=votes.get)
            if dev_cl == pred_cl:
                # print(f"Passed {dev_fn} Pred class: {pred_cl}")
                correct += 1
            else:
                # print(f"Failed {dev_fn} -- Expected {dev_cl}, predicted {pred_cl}")
                _ = 1
            total += 1
    acc = correct/total
    return acc

def distfun_choice(pr_type):
    if pr_type == 'synth':
        return euclidean
    elif pr_type == 'image':
        return euclidean
    elif pr_type == 'digit':
        return euclidean    # previously dtw
    elif pr_type == 'char':
        return euclidean    # previously dtwangle
    else:
        print(f"Problem type '{pr_type}' not present")
        return None

if __name__ == "__main__":
    
    algos = ['pca']
    # algos = ['raw', 'pca', 'lda']
    pr_types = ['synth', 'image', 'digit', 'char']
    
    resize_methods = ['resample']
    # resize_methods = ['resample', 'pad_length']
    multhread = False

    if multhread:
        # Multithread

        for pr in pr_types:
            for algo in algos:
                print(f"\n Starting KNN testing on {algo} {pr} ... \n")
                with open(f'./Data/Pickles/{pr}_{algo}_train.pkl', 'rb') as f:
                    train_feats = pkl.load(f)
                
                with open(f'./Data/Pickles/{pr}_{algo}_dev.pkl', 'rb') as f:
                    dev_feats = pkl.load(f)

                dev_classes = sorted(list(dev_feats.keys()))
                thread_lst = []
                dev_fn_ct = []
                for dev_cl in dev_classes:
                    thread_lst.append([ThreadWithReturnValue(target=knn, args=(train_feats, {dev_cl:dev_feats[dev_cl]}, distfun_choice(pr))), len(list(dev_feats[dev_cl]))])
                    
                for t in thread_lst:
                    t[0].start()

                acc_tot = 0
                for i in range(len(thread_lst)):
                    t = thread_lst[i]
                    acc = t[0].join()
                    print(f"Acc on {algo} {pr} {dev_classes[i]}: {acc}")
                    acc_tot += t[1]*acc
                
                acc_tot /= sum([v[1] for v in thread_lst])
                
                print(f"\n Overall Acc on {algo} {pr}: {acc_tot}\n")
                # acc = knn(train_feats, dev_feats, distfun_choice(pr))
                
                print(f"\n Finished KNN testing on {algo} {pr} \n")

    else:
        # Single thread
        # pr_types = ['synth']
        # k_range = [1,2,3]
        k_range = [1,2,3,5,7,10,15,20]
        
            # print(f"\n\nStarting k={k} ... \n\n")
        acc_pr = {k:None for k in pr_types}
        for pr in pr_types:
            acc_v_k_here = []
            for algo in algos:
                for rm in resize_methods:
                    for k in k_range:
                        prefix = f'{pr}_{algo}'
                        if pr in ['digit', 'char']:
                            prefix += f'_{rm}'
                        print(f"\nStarting KNN testing on k={k}, {prefix} ...")
                        with open(f'./Data/Pickles/{prefix}_train.pkl', 'rb') as f:
                            train_feats = pkl.load(f)
                        
                        with open(f'./Data/Pickles/{prefix}_dev.pkl', 'rb') as f:
                            dev_feats = pkl.load(f)
                        
                        acc_tot = knn(train_feats, dev_feats, distfun_choice(pr), k=k)
                        print(f"Overall Acc on k={k}, {prefix} : {acc_tot}")
                        acc_v_k_here.append(acc_tot)
                        print(f"Finished KNN testing on k={k}, {prefix} \n")
            
            acc_pr[pr] = acc_v_k_here.copy()

    print("Acc_pr:", acc_pr)
    # Plot acc vs k for each dataset
    plt.figure()
    for pr in pr_types:
        plt.plot(k_range, acc_pr[pr])
    
    plt.legend(pr_types)
    plt.title('Acc vs k for Different Datasets')
    plt.savefig('./Plots/KNN_Acc_vs_k_all.png')
    plt.show()
