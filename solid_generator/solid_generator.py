# coding: utf-8

# In[167]:


import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt

dim = 32
min_intensity = 0.5
test_set = 300
train_set = 700


def get_solids(dim, min_intensity):
    size, radius = dim, np.random.randint(3, math.ceil(dim / 4))
    radius2 = np.random.randint(3, math.ceil(dim / 4))
    AA = np.zeros((size, size, size))
    BB = deepcopy(AA)

    x0, y0, z0 = np.random.randint(size * 0.25, size * 0.75, size=(3))

    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                dist = np.linalg.norm(np.array([x0, y0, z0]) - np.array([x, y, z]))
                if dist <= radius: AA[x, y, z] = np.random.rand(1) / 2 + min_intensity

    x0, y0, z0 = np.random.randint(size * 0.25, size * 0.75, size=(3))
    for x in range(x0 - radius2, x0 + radius2 + 1):
        for y in range(y0 - radius2, y0 + radius2 + 1):
            for z in range(z0 - radius2, z0 + radius2 + 1):
                deb = radius2 - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                if deb >= 0: BB[x, y, z] = np.random.rand(1) / 2 + min_intensity
    return AA, BB


for i in range(int(train_set / 2)):
    AA, BB = get_solids(dim, min_intensity)
    AA.dump("./data_solid/%d_s.npy" % i)
    BB.dump("./data_solid/%d_o.npy" % (i + 350))

for i in range(int(test_set / 2)):
    AA, BB = get_solids(dim, min_intensity)
    AA.dump("./test_solid/%d_s.npy" % i)
    BB.dump("./test_solid/%d_o.npy" % (i + 150))

# In[168]:


label_train_set = np.zeros(train_set)
label_test_set = np.zeros(test_set)

label_train_set[int(train_set / 2):] = 1
label_test_set[int(test_set / 2):] = 1

label_train_set.dump("./labels_solid/label_train_set.npy")
label_test_set.dump("./labels_solid/label_test_set.npy")

