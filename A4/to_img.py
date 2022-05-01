import numpy as np
import os
import matplotlib.pyplot as plt

traintest = "train"
dataset = {}

imsize = 50

for fname in os.listdir("Data/HandwritingData/a/"+traintest):
    dataset[str(fname)[:-4]] = np.array(imsize * np.loadtxt("Data/HandwritingData/a/"+traintest+"/"+fname)[1:],dtype=np.int8)

images = np.empty([len(dataset),imsize,imsize])

for i,data in enumerate(list(dataset.values())):
    datax = data[::2]
    datay = data[1::2]
    for x,y in zip(datax,datay):
        print(x,y)
        images[i,imsize-y,imsize-x] = 1

for i in images:
    plt.imshow(i)
    plt.show()