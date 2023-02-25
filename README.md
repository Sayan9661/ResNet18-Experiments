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

## Requirements
Pytorch,Numpy,Matplotlib

## Dataset
CIFAR-10