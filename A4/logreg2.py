import numpy as np
import pandas as pd
import pickle as pkl

from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)

def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd

def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []

    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class MulticlassLR:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)

    def get_acc(self, H, y_dev):
        y_pred = self.predict(H).reshape([-1,1])
        print("y_pred shape:", y_pred.shape)
        print("y_pred unique vals:", np.unique(y_pred))
        print("y_dev shape:", y_dev.shape)
        correct = np.sum(y_pred == y_dev)
        tot = y_pred.shape[0]
        acc = correct/tot
        return acc

if __name__ == "__main__":
    algos = ['raw', 'pca', 'lda']
    pr_types = ['synth', 'image']
    

    for pr in pr_types:
        for algo in algos:
            model = MulticlassLR()

            print(f"\n\n Starting LogReg testing on {algo} {pr} ... \n\n")
            with open(f'./Data/Pickles/{pr}_{algo}_train_np_X.pkl', 'rb') as f:
                X_train = pkl.load(f)

            with open(f'./Data/Pickles/{pr}_{algo}_train_np_y.pkl', 'rb') as f:
                y_train = pkl.load(f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev_np_X.pkl', 'rb') as f:
                X_dev = pkl.load(f)
            
            with open(f'./Data/Pickles/{pr}_{algo}_dev_np_y.pkl', 'rb') as f:
                y_dev = pkl.load(f)
            
            if pr == 'synth' and algo == 'raw':
                #X_train = np.hstack([np.sqrt(np.sum(X_train**2, axis=1)).reshape([-1,1]), np.at
                _ = 1

            print("X_train y_train shapes:", X_train.shape, y_train.shape)
            model.fit(X_train, y_train)
            acc_tot = model.get_acc(X_dev, y_dev)

            print(f"\n\n Overall Acc on {algo} {pr}: {acc_tot}\n\n")

            print(f"\n\n Finished LogReg testing on {algo} {pr} \n\n")
            break
        break