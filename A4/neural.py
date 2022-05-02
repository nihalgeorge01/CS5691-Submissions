# Importing of Important Packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
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
    """
    # Artificial Neural Networks On Synthetic Data

    for algos in ["raw","pca","lda"]:
        # Loading Data
        with open(f"Data/Pickles/synth_{algos}_train_np_X.pkl","rb") as f:
            X_train_syn = pickle.load(f)
        with open(f"Data/Pickles/synth_{algos}_train_np_y.pkl","rb") as f:
            y_train_syn = np.squeeze(pickle.load(f))
        with open(f"Data/Pickles/synth_{algos}_dev_np_X.pkl","rb") as f:
            X_dev_syn = pickle.load(f)
        with open(f"Data/Pickles/synth_{algos}_dev_np_y.pkl","rb") as f:
            y_dev_syn = np.squeeze(pickle.load(f))
        classifier = make_pipeline(StandardScaler(), nn.MLPClassifier(solver='adam', alpha=1e-6,hidden_layer_sizes=(30, 25, 20), random_state=5, max_iter=1000))

        # Fit the data to X and y
        classifier.fit(X_train_syn,y_train_syn)
        preds = classifier.predict(X_dev_syn)
        errs = preds - y_dev_syn
        mistakes = np.count_nonzero(errs)
        print(mistakes)
        acc = 1 - mistakes/(len(y_dev_syn))
        print(f"Accuracy = {acc}")
        x_plot, y_plot = make_meshgrid(X_train_syn[:,0], X_train_syn[:,1])
        contour_plot(classifier,x_plot,y_plot)
        plt.scatter(X_dev_syn[:500,0],X_dev_syn[:500,1])
        plt.scatter(X_dev_syn[500:,0],X_dev_syn[500:,1])
        plt.show()
    """
    # Artificial Neural Networks on Image Data

    for algos in ["raw","pca","lda"]:
        # Loading Data
        with open(f"Data/Pickles/image_{algos}_train_np_X.pkl","rb") as f:
            X_train_img = pickle.load(f)
        with open(f"Data/Pickles/image_{algos}_train_np_y.pkl","rb") as f:
            y_train_img = np.squeeze(pickle.load(f))
        with open(f"Data/Pickles/image_{algos}_dev_np_X.pkl","rb") as f:
            X_dev_img = pickle.load(f)
        with open(f"Data/Pickles/image_{algos}_dev_np_y.pkl","rb") as f:
            y_dev_img = np.squeeze(pickle.load(f))
        classifier = make_pipeline(StandardScaler(), nn.MLPClassifier(solver='adam', alpha=1e-6,hidden_layer_sizes=(30, 25, 20), random_state=5, max_iter=1000))

        # Fit the data to X and y
        classifier.fit(X_train_img,y_train_img)
        preds = classifier.predict(X_dev_img)
        errs = preds - y_dev_img
        mistakes = np.count_nonzero(errs)
        print(mistakes)
        acc = 1 - mistakes/(len(y_dev_img))
        print(f"Accuracy = {acc*100:.2f}%")
        """
        x_plot, y_plot = make_meshgrid(X_train_img[:,0], X_train_img[:,1])
        contour_plot(classifier,x_plot,y_plot)
        plt.scatter(X_dev_img[:500,0],X_dev_img[:500,1])
        plt.scatter(X_dev_img[500:,0],X_dev_img[500:,1])
        plt.show()
        """