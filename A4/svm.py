# Importing of Important Packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from knn import get_posts_list_from_scores, roc_det

# Creating the required utility functions

def contour_plot(classifier, x2d, y2d):
    z_plot = classifier.predict(np.c_[x2d.ravel(),y2d.ravel()]).reshape(x2d.shape)
    plt.contourf(x2d,y2d,z_plot)
def make_meshgrid(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

C = 1e7
kernel = "rbf"

if __name__ == "__main__":
    algos = ["raw","pca","lda"]
    pr_types = ['synth', 'image']
    # Support Vector Machines On Synthetic Data
    for pr in pr_types:
        posts_ll = {}
        cases = []
        for algo in algos:
            prefix = f'{pr}_{algo}'
            # Loading Data
            with open(f"Data/Pickles/{pr}_{algo}_train_np_X.pkl","rb") as f:
                X_train = pickle.load(f)
            with open(f"Data/Pickles/{pr}_{algo}_train_np_y.pkl","rb") as f:
                y_train = np.squeeze(pickle.load(f))
            with open(f"Data/Pickles/{pr}_{algo}_dev_np_X.pkl","rb") as f:
                X_dev = pickle.load(f)
            with open(f"Data/Pickles/{pr}_{algo}_dev_np_y.pkl","rb") as f:
                y_dev = np.squeeze(pickle.load(f))
            classifier = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel=kernel, gamma="auto", probability=True))

            # Fit the data to X and y
            classifier.fit(X_train,y_train)
            preds = classifier.predict(X_dev)
            scores = classifier.predict_proba(X_dev)
            posts = get_posts_list_from_scores(scores,y_dev)
            posts_ll[prefix] = posts.copy()
            cases.append(prefix)
            
            errs = preds - y_dev
            mistakes = np.count_nonzero(errs)
            #print(f"Misclassifications: {mistakes} in {len(preds)}")
            acc = 1 - mistakes/(len(y_dev))
            print(f"{algo}'s accuracy on {pr} = {acc*100:.2f}%")
            if False:#pr == "synth":
                x_plot, y_plot = make_meshgrid(X_train[:,0], X_train[:,1])
                contour_plot(classifier,x_plot,y_plot)
                plt.scatter(X_dev[:500,0],X_dev[:500,1])
                plt.scatter(X_dev[500:,0],X_dev[500:,1])
                plt.show()
        roc_det(posts_ll, cases, pr, extra=f"SVM_{C}_{kernel}")
        plt.show()
    pr_types = ["char", "digit"]
    rect_types = ["pad_length", "resample"]
    for pr in pr_types:
        posts_ll = {}
        cases = []
        for algo in algos:
            for rect in rect_types:
                # Loading Data
                with open(f"Data/Pickles/{pr}_{algo}_{rect}_train_np_X.pkl","rb") as f:
                    X_train = pickle.load(f)
                with open(f"Data/Pickles/{pr}_{algo}_{rect}_train_np_y.pkl","rb") as f:
                    y_train = np.squeeze(pickle.load(f))
                with open(f"Data/Pickles/{pr}_{algo}_{rect}_dev_np_X.pkl","rb") as f:
                    X_dev = pickle.load(f)
                with open(f"Data/Pickles/{pr}_{algo}_{rect}_dev_np_y.pkl","rb") as f:
                    y_dev = np.squeeze(pickle.load(f))
                classifier = make_pipeline(StandardScaler(), svm.SVC(C=C, kernel=kernel, gamma="auto", probability=True))

                # Fit the data to X and y
                classifier.fit(X_train,y_train)
                preds = classifier.predict(X_dev)
                errs = preds - y_dev
                mistakes = np.count_nonzero(errs)
                #print(f"Misclassifications: {mistakes} in {len(preds)}")
                acc = 1 - mistakes/(len(y_dev))
                s = "padding" if rect=="pad_length" else "resampling"
                print(f"{algo}'s accuracy on {pr} with {s} = {acc*100:.2f}%")

                scores = classifier.predict_proba(X_dev)
                posts_list = get_posts_list_from_scores(scores, y_dev)
                posts_ll[f'{pr}_{algo}_{rect}'] = posts_list.copy()
                cases.append(f'{pr}_{algo}_{rect}')
        roc_det(posts_ll, cases, pr, extra=f"SVM_{C}_{kernel}")
        plt.show()