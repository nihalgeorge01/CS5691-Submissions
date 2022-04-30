
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

# Constants
INF = 9999999999

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(v):
    return np.exp(-v)/np.sum(np.exp(-v), axis=1)

def label2onehot(y_l):
    classes = sorted(list(np.unique(y_l)))
    cl_ct = len(classes)
    M = y_l.shape[0]

    cl2ind = {classes[i]:i for i in range(cl_ct)}

    y_o = np.zeros([M, cl_ct])
    for i in range(M):
        y_o[i, cl2ind[y_l[i]]] = 1
    
    return y_o

class MulticlassLR():
    def __init__(self):
        self.W = None

    def fit(self, X_train, y_train, iter_ct = 1000, lr = 0.1, reg = 0.001):
        # Make one-hot encoding for classes (for OvA crossentropy)
        y_train_o = label2onehot(y_train)
        cl_ct = y_train_o.shape[1]
        M = X_train.shape[0]

        # Define hypothesis Wx. Zero initialisation
        W = np.zeros([X_train.shape[1], cl_ct])
        Z = - X_train @ W
        Z_s = softmax(Z)
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
        
        self.W = W

    def predict(self, X_dev):
        Z = - X_dev @ self.W
        Z_s = softmax(Z)
        y_pred = np.argmax(Z_s, axis=1)
        return y_pred

    def get_acc(self, X_dev, y_dev):
        y_pred = self.predict(X_dev)
        correct = np.sum(y_pred == y_dev)
        tot = y_pred.shape[0]
        acc = correct/tot
        return acc

if __name__ == "__main__":
    pass

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# onehot_encoder = OneHotEncoder(sparse=False)

# def loss(X, Y, W):
#     """
#     Y: onehot encoded
#     """
#     Z = - X @ W
#     N = X.shape[0]
#     loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
#     return loss

# def gradient(X, Y, W, mu):
#     """
#     Y: onehot encoded 
#     """
#     Z = - X @ W
#     P = softmax(Z, axis=1)
#     N = X.shape[0]
#     gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
#     return gd

# def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
#     """
#     Very basic gradient descent algorithm with fixed eta and mu
#     """
#     Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
#     W = np.zeros((X.shape[1], Y_onehot.shape[1]))
#     step = 0
#     step_lst = [] 
#     loss_lst = []
#     W_lst = []
 
#     while step < max_iter:
#         step += 1
#         W -= eta * gradient(X, Y_onehot, W, mu)
#         step_lst.append(step)
#         W_lst.append(W)
#         loss_lst.append(loss(X, Y_onehot, W))

#     df = pd.DataFrame({
#         'step': step_lst, 
#         'loss': loss_lst
#     })
#     return df, W

# class Multiclass:
#     def fit(self, X, Y):
#         self.loss_steps, self.W = gradient_descent(X, Y)

#     def loss_plot(self):
#         return self.loss_steps.plot(
#             x='step', 
#             y='loss',
#             xlabel='step',
#             ylabel='loss'
#         )

#     def predict(self, H):
#         Z = - H @ self.W
#         P = softmax(Z, axis=1)
#         return np.argmax(P, axis=1)
    
# X = load_iris().data
# Y = load_iris().target

# # fit model
# model = Multiclass()
# model.fit(X, Y)

# # plot loss
# model.loss_plot()

# # predict 
# model.predict(X)

# # check the predicted value and the actual value
# model.predict(X) == Y
