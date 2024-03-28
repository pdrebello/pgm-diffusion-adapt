""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

from PIL import Image


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import torchvision.transforms as transforms
from utils import toggle_grad

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
import torch.utils.data as data

import os
import wandb
os.environ["http_proxy"] = "http://proxy.cmu.edu:3128"
os.environ["https_proxy"] = "http://proxy.cmu.edu:3128"
os.environ['WANDB_API_KEY'] = '3c85f0f8bd34c1afe1b2d8d0c0a9e43513feebf3'

data_dict = {'mnist_m': 'MNIST_M', 'mnist': 'MNIST', 'svhn': 'SVHN','syn_digits':'SYN_DIGITS','usps':'USPS', 'sign':'SIGN','syn_sign':'SYN_SIGN', 'sign64':'SIGN64','syn_sign64':'SYN_SIGN64'}

print("GPU Av: {}".format(torch.cuda.device_count()))

from train import test_acc, source_domain_numpy, domain_test_numpy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




def train(model, criterion, optimizer, trainloader):

    #for epoch in range(2):  # loop over the dataset multiple times
    correct = 0.0
    total = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, domain = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Finished Training')
    return running_loss, float(correct)/total * 100

def val(model, criterion, optimizer, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return float(correct)/total * 100
    
def run(config):
    utils.seed_rng(config['seed'])

    #transforms_train = transforms.Compose([transforms.Resize(config['resolution']),transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    transforms_test = transforms.Compose([transforms.Resize(config['resolution']),transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    transforms_train = transforms.Compose(
                    [
                    #transforms.ToPILImage(),
#                     transforms.RandomRotation(30),
                    transforms.Resize(config['resolution']),
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[train_mean], std=[train_std]),
                    transforms.Normalize([0.5], [0.5])
                    ])
    
    data_set = source_domain_numpy(root=config['base_root'], root_list=config['source_dataset'], transform=transforms_train)
    loaders = torch.utils.data.DataLoader(data_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True,  worker_init_fn=np.random.seed,drop_last=True)
    
    test_set = domain_test_numpy(root= config['base_root'], root_t=config['target_dataset'], transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, worker_init_fn=np.random.seed, drop_last=True)

    print(len(data_set), len(test_set))
    #exit()
    run = wandb.init(
        project="pgm_project", 
        name="train_target_mnist_m_augmented",
        job_type="Train", 
        config=config,
    )
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    for epoch in range(config['num_epochs']):
        loss, accuracy = train(model, criterion, optimizer, loaders)
        print("Train: {} - Loss: {}, Accuracy: {}".format(epoch, loss, accuracy))

        val_accuracy = val(model, criterion, optimizer, test_loader)
        print("Test: {} - Accuracy: {}".format(epoch, val_accuracy))

        wandb_metric = {'epoch':epoch, 'loss':loss, 'accuracy':accuracy,'val_accuracy':val_accuracy}
        wandb.log(wandb_metric)
    wandb.finish()

def main():
    # parse command line and run
    #parser = utils.prepare_parser()
    #config = vars(parser.parse_args())
    #print(config)
    #run(config)
    config = {'base_root':'../data', 'source_dataset':'mnist,svhn,syn_digits', 'target_dataset':'mnist_m', 'batch_size':256,\
             'resolution':28,'num_workers':4, 'seed':1, 'num_epochs': 100}
    #config = {'base_root':'../data', 'source_dataset':'mnist,svhn,syn_digits,mnist_m_synthetic', 'target_dataset':'mnist_m', 'batch_size':256,\
    #         'resolution':28,'num_workers':4, 'seed':1, 'num_epochs': 100}
    #config = {'base_root':'../data', 'source_dataset':'mnist_m_synthetic', 'target_dataset':'mnist_m', 'batch_size':256,\
    #         'resolution':28,'num_workers':4, 'seed':1, 'num_epochs': 100}
    #for source_data in ['mnist,svhn,syn_digits', 'mnist,svhn,syn_digits,mnist_m_synthetic', 'mnist_m_synthetic']:
        #config['source_dataset'] = source_data
        #run(config)
    config['source_dataset'] = 'mnist,svhn,syn_digits,mnist_m_synthetic'
    #config['target_dataset'] = 'svhn'
    run(config)

if __name__ == '__main__':
    main()