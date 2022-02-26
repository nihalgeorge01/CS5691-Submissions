import numpy as np
import matplotlib.pyplot as plt

x_train,y_train = np.loadtxt("./Data/1D/1d_team_29_train.txt").T
plt.figure(1)
plt.scatter(x_train,y_train)
plt.title("Data Values")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("Data.png")
plt.show()

def regr_polycoeffs(x,y,order,reg=0):
    x = np.asarray(x)
    y = np.asarray(y)
    phi = np.asarray([x**i for i in range(order+1)]).T
    phiT_phi_inv = np.linalg.inv(phi.T @ phi + np.diag([reg]*(order+1)))
    coeffs = phiT_phi_inv @ phi.T @ y

    return coeffs[::-1]

plotrange = np.linspace(0,5,1000)
c7 = regr_polycoeffs(x_train,y_train,7,0)
p7 = np.poly1d(c7)
plt.figure(2)
plt.scatter(x_train,y_train)
plt.plot(plotrange, p7(plotrange),'r')
plt.show()

plotrange = np.linspace(0,5,1000)
c10 = regr_polycoeffs(x_train,y_train,10,0)
p10 = np.poly1d(c10)
plt.figure(3)
plt.scatter(x_train,y_train)
plt.plot(plotrange, p10(plotrange),'g')
plt.show()

plotrange = np.linspace(0,5,1000)
c12 = regr_polycoeffs(x_train,y_train,12,0)
p12 = np.poly1d(c12)
plt.figure(3)
plt.scatter(x_train,y_train)
plt.plot(plotrange, p12(plotrange),'g')
plt.show()