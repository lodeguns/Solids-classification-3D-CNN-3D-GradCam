import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
arr = np.load('conv2.npy')

print(arr)

for s in range(10):
	for i in range(arr.shape[4]):
		print(s, i, np.min(arr[s,:,:,:,i]), np.max(arr[s,:,:,:,i]))

n = int(input("Select the image index (-1 to exit): "))
while n != -1:
	r = raw_input("Select the filter index (empty for next, n to exit): ")
	if r == "":
		i = 0
	elif r != "n":
		i = int(r)
	while r != "n":
		obj = arr[n,:,:,:,i] > np.percentile(arr[n,:,:,:,i], 75)
		print("%.5f " * 5 % tuple(np.percentile(arr[n,:,:,:,i], [0, 25, 50, 75, 100])))
		print(i)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.voxels(obj, facecolors='red', edgecolor='k')
		plt.show()
		r = raw_input("Select the filter index (empty for next, n to exit): ")
		if r == "":
			i += 1
			i %= arr.shape[4]
			continue
		if r == "n":
			break
		i = int(r)
	n = int(input("Select the image index (-1 to exit): "))
			

