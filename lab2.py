import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from ignite.engine import *
from ignite.metrics import *
import time

from model import ResNet18
from model_no_batch import ResNet18_without_Batch



transform_train =  torchvision.transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--optimizer")
parser.add_argument("--datapath")
parser.add_argument("--num_workers")
parser.add_argument("--batchnorm")
args = parser.parse_args()

datapath = './data' # default value of data path

if args.datapath:
    datapath = args.datapath
    print('override default data path by',datapath)


#needed for C7
device = torch.device(args.device)
if args.batchnorm != 'no':
    resnet = ResNet18().to(device)
else:
    resnet = ResNet18_without_Batch().to(device)


#needed for C6
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1,momentum= 0.9,weight_decay= 5e-4)
elif args.optimizer == 'nesterov':
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.1,momentum= 0.9,weight_decay= 5e-4,nesterov=True)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(resnet.parameters())
elif args.optimizer == 'adadelta':
    optimizer = torch.optim.Adadelta(resnet.parameters())





print(f"The parameters taken in arg-parser")
print(f"The optimizer={args.optimizer}, num_workers= {args.num_workers}, device= {args.device}, batchnorm={args.batchnorm}, data path={args.datapath}")




training_data = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train) 
test_data = torchvision.datasets.CIFAR10(root=datapath,  train=False, download=True, transform=transform_test)

data_loader_train = torch.utils.data.DataLoader(training_data,batch_size=128,num_workers=int(args.num_workers))
data_loader_test = torch.utils.data.DataLoader(test_data,batch_size=100,num_workers=int(args.num_workers))


#we have used cross entropy loss as asked in the question
MSELoss = torch.nn.CrossEntropyLoss()


#predefined libraries used for 1% accuracy calculation needed for C6 and C7
def process_function(engine,inputs):
    return inputs # we don't want to change inputs

engine = Engine(process_function)
metric = TopKCategoricalAccuracy(k=1)
metric.attach(engine, 'top_k_accuracy')


#used to measure time in various parts of code
train_time_all = 0
entire_epoch_time = 0
training_loss = 0
load_time_all = 0

# main loading and training below
for epoch in range(5):
    
    
    #start of epoch    
    start_of_epoch = time.perf_counter()

    

    index = 0
    #used to iterate over data
    iterator_data = iter(data_loader_train)
    
    #used for calculating the average loss and accuracy
    acc_sum = 0
    acc_num = 0
    loss_num = 0
    
    #used to measure loading and training time in every epoch
    train_time_epoch = 0
    load_time_per_epoch = 0


    while True:

        #start to load data
        start_load = time.perf_counter()

        try:
            images, labels = next(iterator_data)
        #if there are no more minibatches
        except:
            break

        index += 1
        #transfer to device
        images = images.to(device)
        labels = labels.to(device)

        #data is loaded measure time
        end_load = time.perf_counter()

        load_time_batch = end_load - start_load
        load_time_per_epoch += load_time_batch

        
        #training starts so measure time here
        time_at_train_start = time.perf_counter()

        optimizer.zero_grad()
        pred_out = resnet(images)
        loss_for_preds = MSELoss(pred_out,labels)
        batch_train_loss = loss_for_preds.item()
        # loss_num += batch_train_loss

        loss_for_preds.backward()
        optimizer.step()
    
        training_loss += batch_train_loss

        #end of training so measure time
        time_at_train_end = time.perf_counter()

        train_time_for_minibatch = time_at_train_end - time_at_train_start
        train_time_epoch += train_time_for_minibatch

        #needed for C6 and C7
        state = engine.run([[pred_out, labels]])
        acc_sum += state.metrics['top_k_accuracy']*100*len(labels)
        acc_num += len(labels)
        # print(len(labels))
        loss_num += batch_train_loss
    
    
 #end of epoch
    print('avg acc',acc_sum/acc_num)
    # print(len(labels))
    print('loss:',loss_num)

    end_of_epoch = time.perf_counter()

    time_for_current_epoch = end_of_epoch - start_of_epoch
    
    print(f"Epoch number {epoch} , total time = {time_for_current_epoch}sec , Time to load data= {load_time_per_epoch}sec , Time to train= {train_time_epoch} sec")

    load_time_all += load_time_per_epoch
    train_time_all += train_time_epoch
    entire_epoch_time += time_for_current_epoch
    
    print('\n')
    print('\n')

print(f"total time for 5 epochs= {entire_epoch_time} sec, total time to load data: {load_time_all} sec, total time to train: {train_time_all}sec")

print('\n')
print('\n')


    




