import numpy as np
import math
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#load datasets
train_set_data = np.load("train_set_data.npy")  # training set
arr            = np.load('conv2.npy')           # feature map lvl 1 ( last batch last epoch )
A              = np.load('conv3.npy')           # feature map lvl 2 ( last batch last epoch )
grads          = np.load('grads.npy')[0]        # grads             ( last batch last epoch )

#params
nimg    = 1
sfactor = 2 # for the interpolation of gradients from lvl 2 to lvl 1
feat    = 20  # feature map considered for the plot.


def plot3d(arr):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(arr, facecolors='red', edgecolor='k')
    plt.show()


print(grads.shape)

alpha = np.mean(grads, axis=(1,2,3))

A0 = A[nimg,:,:,:,:]
alpha0 = alpha[nimg,:]

acc = np.zeros(A.shape[1:4])
for i in range(alpha0.shape[0]):
    acc += alpha0[i] * A0[:,:,:,i]

#acc[acc<0] = 0  # relu

#acc = acc * (acc > 0) #another version of relu

print(acc.shape)
print("GradCAM")
plot3d(acc)


new_mat = np.zeros([acc.shape[0] * sfactor, acc.shape[1] * sfactor, acc.shape[2] * sfactor])
print(new_mat.shape)
            


arr = arr[nimg,:,:,:,feat] > np.percentile(arr[nimg,:,:,:,feat], 75)
print("Conv feature map (layer 1)")
plot3d(arr)

selimg = train_set_data[nimg,:,:,:]
print("Input image")
plot3d(selimg)

from scipy.interpolate import RegularGridInterpolator
my_interpolating_function = RegularGridInterpolator((range(16), range(16), range(16)), acc)

pts = []
for x in np.linspace(0, 15, 32):
    for y in np.linspace(0, 15, 32):
        for z in np.linspace(0, 15, 32):
            pts.append([x,y,z])

pts = np.array(pts)
up_acc = my_interpolating_function(pts)

up_acc = up_acc.reshape((32,32,32))
print("Upsampled GradCAM")
plot3d(up_acc)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors

def cuboid_data(center, size=(1,1,1)):
    # code taken from
    # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=alpha)

def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1, cutoff=0, ticks=True):
    # plot a Matrix 
    norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k]) 
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    if data[i,j,k] > cutoff and data[i,j,k] != 0:
                        plotCubeAt(pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha,  ax=ax)



    if cax !=None and ticks:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
        cbar.set_ticks(np.linspace(data.min(),data.max(),10))
        # set the colorbar transparent as well
        cbar.solids.set(alpha=alpha)              


# x and y and z coordinates
x = np.array(range(32))
y = np.array(range(32))
z = np.array(range(32))
data_value = up_acc
print(data_value.shape)

fig = plt.figure(figsize=(10,6))
ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])
ax.set_xlim(0, 31)
ax.set_ylim(0, 31)
ax.set_zlim(0, 31)
ax.set_aspect('equal')

unicolor_img = selimg
unicolor_img[unicolor_img > 0] = 1

plotMatrix(ax, x, y, z, unicolor_img, cmap="jet", cax = ax_cb, alpha=0.06, ticks = False)

plotMatrix(ax, x, y, z, data_value, cmap="jet", cax = ax_cb, alpha=0.06, cutoff=-10000)

print("Heatmap")
plt.savefig("3d_heatmap.png")
plt.show()
