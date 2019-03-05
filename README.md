# Solids-classification-3D-CNN-3D-GradCam
Here we introduce the problem of 3D CNN solids classification (spheres and octahedra).


<img src="https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/img/sfera.png" height="250" width="350"> <img src="https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/img/octahedron.png" height="250" width="350">


Furthermore, we implemented a 3D GradCam model, in order to underline the most relevant 3D volumes of the original image that are useful for the classification.


### Solid Generator
This script generates spheres and octahedra (for the test and training set) depending on the max size of the solid and its diameter. The positions (x,y,z) in the space NxNxN are given randomly and the voxels intensities inside the solids are chosen randomly in a fixed interval.

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

The model is so simple that reach the 100% of accuracy on the test set in the 4th epoch. 

### Test the features
Once you run the model, in the training are saved all the numpy matrices of gradients, weights, etc..
these one could be checked with the script test_feature.py. Whit this script you can walk around the filters and check in which way the CNN works.

``` 
python test_feature.py
```
Such for example here are shown, for a sphere and an octahedron, 2 CNN filters given in the first convolutional layer:

<img src="https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/myimage.gif" height="250" width="250"> <img src="https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/myimage2.gif" height="250" width="250">

### 3D Grad Cam

With simple set-ups is possible to visualize (positive-negative) gradients of the last layer on a 3D space. 
``` 
python plot_gradcam.py
```
<img src="https://github.com/lodeguns/Solids-classification-3D-CNN-3D-GradCam/blob/master/gradcam.gif" height="350" width="550">



@NeuroneLab - University of Salerno

Francesco Bardozzo (fbardozzo@unisa.it)

Gioele Ciaparrone  (gciaparrone@unisa.it)

Mattia Delli Priscoli (mdellipriscoli@unisa.it)
