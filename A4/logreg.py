
# TODO Logistic Regression
# 
# Load features from file, then apply model
# Try diff params, give accuracies, misclassification rates for each dataset
# Plot confusion matrix, ROC, DET
# Do not use inbuilt library for Logistic Regression
# Do on three forms of data -- normal, after PCA, after LDA

# feature type - normal, LDA, PCA
# dataset - Image, Synthetic, IsolatedDigits, HandwrittenCharacters

# Imports
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from knn import get_posts_list_from_scores, roc_det
# from scipy.special import softmax

# Constants
INF = 9999999999

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(v, axis=1):
    return np.exp(v)/np.sum(np.exp(v), axis=axis).reshape([-1,1])

def polyfeatures(X, deg=6):
    # Get polynomial features upto some degree
    feat_ct = X.shape[1]
    for j in range(feat_ct):
        col_here = X[:,j]
        for d in range(2, deg):
            X = np.hstack((X, (col_here**d).reshape([-1,1])))

    return X

def label2onehot(y_l):
    classes = sorted(list(np.unique(y_l)))
    cl_ct = len(classes)
    M = y_l.shape[0]

    cl2ind = {classes[i]:i for i in range(cl_ct)}

    y_o = np.zeros([M, cl_ct])
    for i in range(M):
        y_o[i, cl2ind[y_l[i,0]]] = 1
    
    print("y_o shape:", y_o.shape)
    return y_o

class MulticlassLR():
    def __init__(self):
        self.W = None

    def fit(self, X_train, y_train, iter_ct = 1000, lr = 0.001, reg = 0):
        # Make one-hot encoding for classes (for OvA crossentropy)
        y_train_o = label2onehot(y_train)
        cl_ct = y_train_o.shape[1]
        M = X_train.shape[0]

        # Define hypothesis Wx. Zero initialisation
        W = np.zeros([X_train.shape[1], cl_ct])
        Z = - X_train @ W
        # Z_s = softmax(Z)
        Z_s = softmax(Z, axis=1)
        loss_here = (1/M) * (np.trace(X_train @ W @ y_train_o.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        losses = [loss_here]
        # Gradient descent iterations
        for i in range(iter_ct):
            grad = (1/M) * (X_train.T @ (y_train_o-Z_s)) + 2*reg*W # L2 reg
            W -= lr*grad
            loss_here = (1/M) * (np.trace(X_train @ W @ y_train_o.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
            losses.append(loss_here)
            Z = - X_train @ W
            Z_s = softmax(Z)
        
        print("losses:", losses[:10])
        self.W = W

    def predict(self, X_dev):
        Z = - X_dev @ self.W
        Z_s = softmax(Z)
        y_pred = np.argmax(Z_s, axis=1).reshape([-1,1])
        return y_pred, Z_s

    def get_acc(self, X_dev, y_dev):
        y_pred, Z_s = self.predict(X_dev)
        # print("y_pred shape:", y_pred.shape)
        # print("y_pred unique vals:", np.unique(y_pred))
        # print("y_dev shape:", y_dev.shape)
        posts_list = get_posts_list_from_scores(Z_s, y_dev)
        correct = np.sum(y_pred == y_dev)
        tot = y_pred.shape[0]
        acc = correct/tot
        return acc, posts_list

if __name__ == "__main__":
    # algos = ['raw', 'pca', 'lda']
    algos = ['pca']
    # pr_types = ['image']
    # resize_methods = ['resample', 'pad_length']
    resize_methods = ['resample']
    pr_types = ['synth', 'image', 'char', 'digit']

    # lr_range = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
    lr_range = [1e-2]
    acc_pr = {k:None for k in pr_types}
    for pr in pr_types:
        acc_v_p_here = []
        posts_ll = {}
        cases = []
        for algo in algos:
            for resize_method in resize_methods:
                for lr in lr_range:
                    model = MulticlassLR()
                    prefix = f'{pr}_{algo}'
                    if pr in ['digit', 'char']:
                        prefix += f'_{resize_method}'
                    print(f"\n\n Starting LogReg testing on {prefix} ... \n\n")
                    with open(f'./Data/Pickles/{prefix}_train_np_X.pkl', 'rb') as f:
                        X_train = pkl.load(f)

                    with open(f'./Data/Pickles/{prefix}_train_np_y.pkl', 'rb') as f:
                        y_train = pkl.load(f)
                    
                    with open(f'./Data/Pickles/{prefix}_dev_np_X.pkl', 'rb') as f:
                        X_dev = pkl.load(f)
                    
                    with open(f'./Data/Pickles/{prefix}_dev_np_y.pkl', 'rb') as f:
                        y_dev = pkl.load(f)
                    
                    if pr == 'synth' and algo == 'raw':
                        #X_train = np.hstack([np.sqrt(np.sum(X_train**2, axis=1)).reshape([-1,1]), np.at
                        _ = 1

                    # print("X_train y_train shapes:", X_train.shape, y_train.shape)
                    deg = 2
                    X_train, X_dev = polyfeatures(X_train, deg), polyfeatures(X_dev, deg)
                    model.fit(X_train, y_train, lr=3e-3, iter_ct=5000)
                    acc_tot, posts = model.get_acc(X_dev, y_dev)
                    preds, _ = model.predict(X_dev=X_dev)
                    conf_mat = confusion_matrix(y_dev, preds)
                    plot = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
                    plot.plot()
                    plt.savefig(f"Plots/confmat_log_{pr}_{algo}")
                    plt.show()
                    posts_ll[prefix+f'{lr}'] = posts.copy()
                    cases.append(prefix+f'{lr}')
                    acc_v_p_here.append(acc_tot)
                    print(f"\n\n Overall Acc on lr={lr}, {prefix}: {acc_tot}\n\n")

                    print(f"\n\n Finished LogReg testing on lr={lr}, {prefix} \n\n")
                    # if pr not in ['digit', 'char']:
                    #     break
            roc_det(posts_ll, cases, pr, extra='LogReg_poly_testing')
        acc_pr[pr] = acc_v_p_here.copy()

    print("Acc_pr:", acc_pr)
    # Plot acc vs k for each dataset
    # plt.figure()
    # for pr in pr_types:
    #     plt.plot(lr_range, acc_pr[pr])
    
    # plt.legend(pr_types)
    # plt.title('Acc vs Learning Rate for Different Datasets')
    # plt.savefig('./Plots/LogReg_Acc_vs_lr_at_reg_0_iters_1k_all.png')
    # plt.show()