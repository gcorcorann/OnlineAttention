#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class SingleFrame(nn.Module):
    """Single Frame Model."""
    def __init__(self):
        super().__init__()
        #TODO change to pretrained vgg19_bn
        self.cnn = models.vgg11()
        # number of features in fc1
        num_ftrs = self.cnn.classifier[3].in_features
        # remove last two fc layers (+ ReLU and Dropout layers)
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        self.fc = nn.Linear(num_ftrs, 4)

    def forward(self, inp):
        out = self.cnn.forward(inp)
        out = self.fc(out)
        return out

def main():
    """Main Function."""

    # hyper-parameters
    gpu = torch.cuda.is_available()
    num_epochs = 20
    sequence_len = 5
    batch_size = 2
    input_size = (224,224)

    # create network
    net = VGGNet()
    print(net)
    if gpu:
        net = net.cuda()

    # create inputs and targets
    inputs = torch.randn(sequence_len, batch_size, 3, *input_size)
    targets = torch.LongTensor(batch_size).random_(4)
    print('inputs:', inputs.shape)
    print('targets:', targets.shape)
    # store in Variables
    if gpu:
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
    else:
        inputs = Variable(inputs)
        targets = Variable(targets)

    # training
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('Epoch', epoch+1)
        print('-' * 10)
        
        # each frame in sequence 
        batch_loss = 0
        for inp in inputs:
            # pass through network
            output = net.forward(inp)
            # compute loss
            loss = criterion(output, targets)
            batch_loss += loss

        batch_loss /= sequence_len
        print('Batch Loss:', batch_loss)

        # clear existing gradients
        net.zero_grad()

        # back-prop
        batch_loss.backward()

        # update weights
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
        optimizer.step()

if __name__ == '__main__':
    main()
