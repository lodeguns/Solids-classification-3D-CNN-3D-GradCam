# Solids-classification-3D-CNN-3D-GradCam
Here we introduce the problem of 3D solids classification (spheres and octahedra) through a 3D CNN.

The source is written in tensorflow/python.

We implemented a 3D GradCam model, in order to underline the most relevant 3D volumes of the original image that are useful for the classification.


### Solid Generator
This script generates solids and octahedra (for the test and training set) depending on the max size of the solid and its diameter. The positions (x,y,z) in the space NxNxN are given randomly and the voxels intensities inside the solid are chosen randomly.

``` 
python2 solid_generator.py
```
Set in your experiment the folders: data_solid, test_solid, labels_solid.


### Model and running
the model and the training is written in the script v0.3 on a Jupyter Notebook.

We have done the experiment on a NVIDIA GeForce 1070 and with this environment:
``` 
Tensorflow version: 1.8.0
Python version: 3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, 19:16:44) 
[GCC 7.3.0]
``` 


Set in your experiment the folders: data_solid, test_solid, labels_solid.

@NeuroneLab - University of Salerno

Francesco Bardozzo (fbardozzo@unisa.it)

Gioele Ciaparrone  (gciaparrone@unisa.it)

Mattia Delli Priscoli (mdellipriscoli@unisa.it)
