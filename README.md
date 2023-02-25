# ResNet18-Experiments

This was done as part of High Performance Machine Learning course.

## Aim/Motivation
Test the performance(Measure the time taken) of ResNet-18 with varying parameters:
Parameters:<br>

<ul>
<li>Device: CPU or GPU</li>
<li>Numworkers</li>
<li>Type of optimizer
<ul>
<li>SGD:simple stochastic gradient descent</li>
<li>Nesterov</li>
<li>Adam</li>
<li>Adagrad</li>
<li>AdaDelta</li>
</ul>
</li>
<li>BatchNorm:yes/no</li>
</ul>

Example command to run file with above arguments.
py lab2.py --device cuda --optimizer sgd --num_workers 8 --batchnorm no

## Requirements
Pytorch,Numpy,Matplotlib

## outputs
It will give you:<br>
<ul>
<li>Time per epoch</li>
<li>Time for total epochs</li>
</ul>

Time Measuments
<ul>
<li>Total time</li>
<li>Time to load data</li>
<li>Time to train</li>
</ul>


## Dataset
CIFAR-10
