# Importing of Important Packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from knn import get_posts_list_from_scores, roc_det
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Creating the required utility functions

def contour_plot(classifier, x2d, y2d):
    z_plot = classifier.predict(np.c_[x2d.ravel(),y2d.ravel()]).reshape(x2d.shape)
    plt.contourf(x2d,y2d,z_plot)
def make_meshgrid(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

optimizer = "adam"
alpha = 1e-6
hidden_layer = (30,25,20)

if __name__ == "__main__":
    
    algos = ["raw","pca","lda"]
    pr_types = ["synth", "image"]
    
    for pr in pr_types:
        posts_ll = {}
        cases = []
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
            classifier = make_pipeline(StandardScaler(), nn.MLPClassifier(solver=optimizer, alpha=alpha, hidden_layer_sizes=hidden_layer, random_state=5, max_iter=1000))

            # Fit the data to X and y
            classifier.fit(X_train,y_train)
            preds = classifier.predict(X_dev)
            errs = preds - y_dev
            mistakes = np.count_nonzero(errs)
            #print(f"Misclassifications: {mistakes} in {len(preds)}")
            acc = 1 - mistakes/(len(y_dev))
            print(f"{algo}'s accuracy on {pr} = {acc*100:.2f}%")
            conf_mat = confusion_matrix(y_dev, preds)
            plot = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
            plot.plot()
            plt.savefig(f"Plots/confmat_ANN_{pr}_{algo}")
            plt.show()
            if False:#pr == "synth":
                x_plot, y_plot = make_meshgrid(X_train[:,0], X_train[:,1])
                contour_plot(classifier,x_plot,y_plot)
                plt.scatter(X_dev[:500,0],X_dev[:500,1])
                plt.scatter(X_dev[500:,0],X_dev[500:,1])
                plt.show()
            scores = classifier.predict_proba(X_dev)
            posts_list = get_posts_list_from_scores(scores, y_dev)
            posts_ll[f'{pr}_{algo}'] = posts_list.copy()
            cases.append(f'{pr}_{algo}')
        roc_det(posts_ll, cases, pr, extra=f"ANN_{optimizer}_{alpha}")
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
                classifier = make_pipeline(StandardScaler(), nn.MLPClassifier(solver=optimizer, alpha=alpha, hidden_layer_sizes=hidden_layer, random_state=5, max_iter=1000))

                # Fit the data to X and y
                classifier.fit(X_train,y_train)
                preds = classifier.predict(X_dev)
                errs = preds - y_dev
                mistakes = np.count_nonzero(errs)
                #print(f"Misclassifications: {mistakes} in {len(preds)}")
                acc = 1 - mistakes/(len(y_dev))
                s = "padding" if rect=="pad_length" else "resampling"
                print(f"{algo}'s accuracy on {pr} with {s} = {acc*100:.2f}%")
                conf_mat = confusion_matrix(y_dev, preds)
                plot = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
                plot.plot()
                plt.savefig(f"Plots/confmat_ANN_{pr}_{algo}_{rect}")
                plt.show()
                scores = classifier.predict_proba(X_dev)
                posts_list = get_posts_list_from_scores(scores, y_dev)
                posts_ll[f'{pr}_{algo}_{rect}'] = posts_list.copy()
                cases.append(f'{pr}_{algo}_{rect}')
        roc_det(posts_ll, cases, pr, extra=f"ANN_{optimizer}_{alpha}")
        plt.show()