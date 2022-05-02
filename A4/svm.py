# Importing of Important Packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Creating the required utility functions

def contour_plot(classifier, x2d, y2d):
    z_plot = classifier.predict(np.c_[x2d.ravel(),y2d.ravel()]).reshape(x2d.shape)
    plt.contourf(x2d,y2d,z_plot)
def make_meshgrid(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

if __name__ == "__main__":
    algos = ["raw","pca","lda"]
    pr_types = ['synth', 'image', 'char', 'digit']
    # Support Vector Machines On Synthetic Data
    for pr in pr_types:
        for algo in algos:
            # Loading Data
            with open(f"Data/Pickles/{pr}_{algo}_train_np_X.pkl","rb") as f:
                X_train = pickle.load(f)
            with open(f"Data/Pickles/{pr}_{algo}_train_np_y.pkl","rb") as f:
                y_train = np.squeeze(pickle.load(f))
            with open(f"Data/Pickles/{pr}_{algo}_dev_np_X.pkl","rb") as f:
                X_dev = pickle.load(f)
            with open(f"Data/Pickles/{pr}_{algo}_dev_np_y.pkl","rb") as f:
                y_dev = np.squeeze(pickle.load(f))
            classifier = make_pipeline(StandardScaler(), svm.SVC(C=1e7, kernel="rbf", gamma="auto"))

            # Fit the data to X and y
            classifier.fit(X_train,y_train)
            preds = classifier.predict(X_dev)
            errs = preds - y_dev
            mistakes = np.count_nonzero(errs)
            print(f"Misclassifications: {mistakes}")
            acc = 1 - mistakes/(len(y_dev))
            print(f"{algo}'s accuracy on {pr} = {acc*100:.2f}%")
            if pr == "synth":
                x_plot, y_plot = make_meshgrid(X_train[:,0], X_train[:,1])
                contour_plot(classifier,x_plot,y_plot)
                plt.scatter(X_dev[:500,0],X_dev[:500,1])
                plt.scatter(X_dev[500:,0],X_dev[500:,1])
                plt.show()