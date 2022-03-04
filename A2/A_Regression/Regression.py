from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(0)

x_train,y_train = np.loadtxt("./Data/1D/1d_team_29_train.txt").T
x_dev,y_dev = np.loadtxt("./Data/1D/1d_team_29_dev.txt").T

plt.figure()
plt.scatter(x_train,y_train)
plt.title("Data Values")
plt.xlabel("x")
plt.ylabel("y")
#plt.savefig("Data.png")
#plt.show()
plt.close()

# Regression Function
# regul=0 for normal Least Squares Linear Regression
# regul>0 for Ridge Regression

def ls_regr(x,y,order,regul=0):
    x = np.asarray(x)
    y = np.asarray(y)
    phi = np.asarray([x**i for i in range(order+1)]).T
    phiT_phi_inv = np.linalg.inv(phi.T @ phi + np.diag([regul]*(order+1)))
    coeffs = phiT_phi_inv @ phi.T @ y
    return coeffs[::-1]

def plotter(x,y,order,regul=0,title=""):
    plotrange = np.linspace(0,5,1000)
    wid = len(x)
    coeffs = ls_regr(x,y,order=order,regul=regul)
    poly = np.poly1d(coeffs)
    plt.figure()
    plt.scatter(x,y)
    plt.plot(plotrange,poly(plotrange),'r')
    if title=="":
        plt.title(f'{order} degree polynomial fit')
    else:
        plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'./Plots/poly{order}_{regul}_{wid}.png')
    #plt.show()
    plt.close()

plotrange = np.linspace(0,5,1000)

# Least Squares Regression (no Regularization)


for i in [0,1,5,7,10]:
    for j in [10,20,50,100,200]:
        myrange = np.random.choice(range(0,200),size=(j,)) # choose j samples out of the range 0-199
        x_in = x_train[myrange] # sample the x values at those indices
        y_in = y_train[myrange] # sample the y values at those indices
        plotter(x_in,y_in,i)    # plot them!


# Ridge Regression (Regularization Parameter non-zero)

for i in [0,1,5,7,10]:
    for j in np.logspace(-9,0,10):
        plotter(x_train,y_train,i,j)    # plot them for different regularization parameters


# Scatter Plot of best Performing Model vs Ground Truth

# Training set
fig = plt.figure()
ax = fig.add_subplot(111)
c10 = ls_regr(x_train,y_train,10,1e-5)
p10 = np.poly1d(c10)
ax.set_aspect('equal')
plt.title(r"Model Output vs Target Output on Order 10 model with $\lambda = 10^{-5}$")
plt.xlabel("Target Output on Training Set")
plt.ylabel("Predicted Output on Training Set")
plt.scatter(y_train,p10(np.linspace(0,5,200)))
plt.savefig("ModelvsTarget_train.png")
plt.show()
plt.close()

# Dev Set
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
c10_d = ls_regr(x_dev,y_dev,10,1e-5)
p10_d = np.poly1d(c10_d)
plt.title(r"Model Output vs Target Output on Order 10 model with $\lambda = 10^{-5}$")
plt.xlabel("Target Output on Dev Set")
plt.ylabel("Predicted Output on Dev Set")
plt.scatter(y_dev,p10_d(np.linspace(0,5,200)))
plt.savefig("ModelvsTarget_dev.png")
plt.show()
plt.close()

# Errors
def error_func(y,poly):
    err = np.sum((poly(np.linspace(0,5,200)) - y)**2)
    RMSE = np.sqrt(err/len(y))
    return RMSE

errs_train = [0]*11
errs_dev = [0]*11

for i in range(10+1):
    c_train = ls_regr(x_train,y_train,i,1e-5)
    p_train = np.poly1d(c_train)
    c_dev = ls_regr(x_train,y_train,i,1e-5)
    p_dev = np.poly1d(c_dev)
    errs_train[i] = error_func(y_train,p_train)
    errs_dev[i] = error_func(y_dev,p_dev)

plt.figure()
plt.title("RMS Error vs Model Complexity")
plt.plot(np.arange(11),errs_train,'-x',label="Training Set")
plt.plot(np.arange(11),errs_dev,'-o',label="Development Set")
plt.xlabel("Order of Polynomial")
plt.ylabel("RMS Error")
plt.legend()
plt.savefig("RMS_1D.png")
plt.show()

errs_train_reg = [0]*15
errs_dev_reg = [0]*15

cnt = 0
for i in np.logspace(-9,4,15):
    c_train = ls_regr(x_train,y_train,10,i)
    p_train = np.poly1d(c_train)
    c_dev = ls_regr(x_train,y_train,10,i)
    p_dev = np.poly1d(c_dev)
    errs_train_reg[cnt] = error_func(y_train,p_train)
    errs_dev_reg[cnt] = error_func(y_dev,p_dev)
    cnt+=1

plt.figure()
plt.title("RMS Error vs Regularization Parameter")
plt.plot(np.logspace(-9,4,15),errs_train_reg,'-x',label="Training Set")
plt.plot(np.logspace(-9,4,15),errs_dev_reg,'-o',label="Development Set")
plt.xlabel(r"Regularization Parameter $\lambda$")
plt.ylabel("RMS Error")
plt.xscale("log")
plt.legend()
plt.savefig("RMS_1D_reg.png")
plt.show()


# 2D Dataset

def ls_regr_2D(x1,x2,y,order,regul=0):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y = np.asarray(y)
    phi = np.vstack(([x1**i for i in range(order+1)],[x2**i for i in range(order+1)])).T
    phiT_phi_inv = np.linalg.pinv(phi.T @ phi + np.diag([regul]*2*(order+1)))
    coeffs = phiT_phi_inv @ phi.T @ y
    return coeffs

def eval2D(x1,x2,coeffs):
    order = len(coeffs)//2
    z = 0
    for i in range(order):
        z += (x1**i)*coeffs[i] + (x2**i)*coeffs[i+order]
    return z

def plotter2D(x1,x2,y,order,regul):
    coeffs = ls_regr_2D(x1,x2,y,order=order,regul=regul)
    y_pred = [0]*len(x1)
    cnt = 0
    for i,j in zip(x1,x2):
        y_pred[cnt] = eval2D(i,j,coeffs)
        cnt+=1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1, x2, y, color="blue")
    ax.plot_trisurf(x1,x2,y_pred)
    fig.savefig(f"./Plots_3D/plot3d_{order}_{regul}_{cnt}.png")
    plt.close()

x1_train_2d, x2_train_2d, y_train_2d = np.loadtxt("./Data/2D/2d_team_29_train.txt").T
x1_dev_2d,x2_dev_2d,y_dev_2d = np.loadtxt("./Data/2D/2d_team_29_dev.txt").T

#plotter2D(x1_train_2d,x2_train_2d,y_train_2d,6,0)


for i in [0,1,5,7,10]:
    for j in [10,20,50,100,200,500,1000]:
        myrange = np.random.choice(range(0,1000),size=(j,)) # choose j samples out of the range 0-199
        x1_in = x1_train_2d[myrange]
        x2_in = x2_train_2d[myrange]
        y_in = y_train_2d[myrange]
        plotter2D(x1_in,x2_in,y_in,i,0)



for i in [0,1,5,7,10]:
    for j in np.logspace(-9,0,10):
        plotter2D(x1_train_2d, x2_train_2d, y_train_2d,i,j)


def error_func_2D(y_pred,y):
    err = np.sum((y_pred - y)**2)
    RMSE = np.sqrt(err/len(y))
    return RMSE

errs_train_2d = [0]*11
errs_dev_2d = [0]*11

for l in range(10+1):
    coeffs_t = ls_regr_2D(x1_train_2d,x2_train_2d,y_train_2d,l)
    coeffs_d = ls_regr_2D(x1_dev_2d,x2_dev_2d,y_dev_2d,l)
    y_pred_2d_t = [0]*len(x1_train_2d)
    y_pred_2d_d = [0]*len(x1_train_2d)
    cnt = 0
    for i,j in zip(x1_train_2d,x2_train_2d):
        y_pred_2d_t[cnt] = eval2D(i,j,coeffs_t)
        cnt+=1
    cnt = 0
    for i,j in zip(x1_dev_2d,x2_dev_2d):
        y_pred_2d_d[cnt] = eval2D(i,j,coeffs_d)
        cnt+=1
    errs_train_2d[l] = error_func_2D(y_pred_2d_t,y_train_2d)
    errs_dev_2d[l] = error_func_2D(y_pred_2d_d,y_dev_2d)

plt.figure()
plt.title("RMS Error vs Model Complexity")
plt.plot(np.arange(11),errs_train_2d,'-x',label="Training Set")
plt.plot(np.arange(11),errs_dev_2d,'-o',label="Development Set")
plt.xlabel("Order of Polynomial")
plt.ylabel("RMS Error")
plt.legend()
plt.savefig("RMS_2D.png")
plt.show()

errs_train_2d_reg = [0]*15
errs_dev_2d_reg = [0]*15

cnt = 0
for l in np.logspace(-9,4,15):
    coeffs_t = ls_regr_2D(x1_train_2d,x2_train_2d,y_train_2d,10,l)
    coeffs_d = ls_regr_2D(x1_dev_2d,x2_dev_2d,y_dev_2d,10,l)
    y_pred_2d_t = [0]*len(x1_train_2d)
    y_pred_2d_d = [0]*len(x1_dev_2d)
    cnt1 = 0
    for i,j in zip(x1_train_2d,x2_train_2d):
        y_pred_2d_t[cnt1] = eval2D(i,j,coeffs_t)
        cnt1+=1
    cnt1 = 0
    for i,j in zip(x1_dev_2d,x2_dev_2d):
        y_pred_2d_d[cnt1] = eval2D(i,j,coeffs_d)
        cnt1+=1
    errs_train_2d_reg[cnt] = error_func_2D(y_pred_2d_t,y_train_2d)
    errs_dev_2d_reg[cnt] = error_func_2D(y_pred_2d_d,y_dev_2d)
    cnt+=1

plt.figure()
plt.title("RMS Error vs Regularization Parameter")
plt.plot(np.logspace(-9,4,15),errs_train_2d_reg,'-x',label="Training Set")
plt.plot(np.logspace(-9,4,15),errs_dev_2d_reg,'-o',label="Development Set")
plt.xlabel(r"Regularization Parameter $\lambda$")
plt.ylabel("RMS Error")
plt.xscale("log")
plt.legend()
plt.savefig("RMS_2D_reg.png")
plt.show()

y_pred_2d_t = [0]*len(x1_train_2d)
y_pred_2d_d = [0]*len(x1_dev_2d)
coeffs_2d_t = ls_regr_2D(x1_train_2d,x2_train_2d,y_train_2d,10,1e-5)
cnt = 0
for i,j in zip(x1_train_2d,x2_train_2d):
    y_pred_2d_t[cnt] = eval2D(i,j,coeffs_2d_t)
    cnt+=1
coeffs_2d_d = ls_regr_2D(x1_dev_2d,x2_dev_2d,y_dev_2d,10,1e-5)
cnt = 0
for i,j in zip(x1_dev_2d,x2_dev_2d):
    y_pred_2d_d[cnt] = eval2D(i,j,coeffs_2d_d)
    cnt+=1


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.title(r"Model Output vs Target Output on Order 10 model with $\lambda = 10^{-5}$")
plt.xlabel("Target Output on Training Set")
plt.ylabel("Predicted Output on Training Set")
plt.scatter(y_train_2d,y_pred_2d_t)
plt.savefig("ModelvsTarget_train_2D.png")
plt.show()
plt.close()

# Dev Set
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
c10_d = ls_regr(x_dev,y_dev,10,1e-5)
p10_d = np.poly1d(c10_d)
plt.title(r"Model Output vs Target Output on Order 10 model with $\lambda = 10^{-5}$")
plt.xlabel("Target Output on Dev Set")
plt.ylabel("Predicted Output on Dev Set")
plt.scatter(y_dev_2d,y_pred_2d_d)
plt.savefig("ModelvsTarget_dev_2D.png")
plt.show()
plt.close()