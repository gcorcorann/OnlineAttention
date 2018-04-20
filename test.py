import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

class SingleStream(nn.Module):
    """Single Stream (Spatial OR Temporal) + LSTM"""
    def __init__(self):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        num_fts = self.cnn.classifier[3].in_features
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        # add lstm layer
        self.lstm = nn.LSTM(num_fts, 512, 2)
        # add linear layer
        self.fc = nn.Linear(512, 4)

    def forward(self, inputs):
        """Forward pass through network.
        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]
        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp).data
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

cap = cv2.VideoCapture('data/dashcam_test.mp4')
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

net = SingleStream()

inputs_list = [torch.zeros(1,3,224,224) for i in range(20)]

while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break

    # use frame as input
    inp = cv2.resize(frame, (224, 224))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp / 255
    inp = (inp - mean) / std
    inp = inp.transpose(2,0,1)
    inp = torch.from_numpy(inp).type(torch.FloatTensor)
    inp = inp.unsqueeze(0)
    inputs_list.append(inp)
    inputs_list.pop(0)
    inputs = Variable(torch.stack((inputs_list)))
    outputs = net.forward(inputs)
    _, pred = torch.max(outputs.data, 1)
    cv2.putText(frame, str(pred), (20,70), cv2.FONT_HERSHEY_SIMPLEX,
            3, (0,0,255), 4)
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
#
#print('Done')