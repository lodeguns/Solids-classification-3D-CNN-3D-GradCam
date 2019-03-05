# Solids-classification-3D-CNN-3D-GradCam
Here we introduce the problem of 3D solids classification (spheres and octahedra) through a 3D CNN.

The source is written in tensorflow/python.

We implemented a 3D GradCam model, in order to underline the most relevant 3D volumes of the original image that are useful for the classification.


### Solid Generator
This script generates solids and octahedra (for the test and training set) depending on the max size of the solid and its diameter. The positions (x,y,z) in the space NxNxN are given randomly and the voxels intensities inside the solid are chosen randomly.

``` 
python solid_generator.py
```
Set in your experiment the folders: data_solid, test_solid, labels_solid.


### Model and Execution
The model and the training is written in the script_v0.3 on a Jupyter Notebook.

We have done the experiment on a NVIDIA GeForce 1070 and with this environment:
``` 
Tensorflow version: 1.8.0
Python version: 3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, 19:16:44) 
[GCC 7.3.0]
``` 


Set in your experiment the folders: data_solid, test_solid, labels_solid.

### Test the features
Once you run the model, in the training are saved all the numpy matrices of gradients, weights, etc..
these one could be checked with the script test_feature.py. Whit this script you can walk around the filters and check in which way the CNN works.

``` 
python test_feature.py
```

### 3D Grad Cam

With simple set-ups is possible to visualize (positive-negative) gradients of the last layer on a 3D space. 
``` 
python plot_gradcam.py
```

![a CNN spheres filter ](https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/myimage.gif =250x250) ![a CNN octahedra filter](https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/myimage2.gif =250x250)
@NeuroneLab - University of Salerno

Francesco Bardozzo (fbardozzo@unisa.it)

Gioele Ciaparrone  (gciaparrone@unisa.it)

Mattia Delli Priscoli (mdellipriscoli@unisa.it)
