# Solids-classification-3D-CNN-3D-GradCam
Here we introduce the problem of 3D solids classification (spheres and octahedra) through a 3D CNN.

The source is written in tensorflow/python.

We implemented a 3D GradCam model, in order to underline the most relevant 3D volumes of the original image that are useful for the classification.


## Solid Generator
This script generate solids and octahedra depending by the max size of the solid and its diameter. The positions (x,y,z) in the space NxNxN are given randomically and the voxels intensities inside the solid are choosen randomically in a specific interval of values.


@NeuroneLab - University of Salerno

Francesco Bardozzo (fbardozzo@unisa.it)

Gioele Ciaparrone  (gciaparrone@unisa.it)

Mattia Delli Priscoli (mdellipriscoli@unisa.it)
