#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loaders
from train import train_network
import model

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    train_path = '/home/gary/datasets/accv/labels/train_data.txt'
    valid_path = '/home/gary/datasets/accv/labels/valid_data.txt'
    batch_size = 2
    sequence_len = 5
    num_workers = 1
    # training parameters
    max_epochs = 2
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()

    # get dataloaders    
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path,
            batch_size, sequence_len, num_workers, gpu)

    # create network and optimizier
    net = model.SingleFrame()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    # train the network
    net, val_acc = train_network(net, dataloaders, dataset_sizes, batch_size, 
            sequence_len, criterion, optimizer, max_epochs, gpu)
    print('Best Validation Acc:', val_acc)
    # save network
    torch.save(net.state_dict(), '../data/single_frame_params.pkl')

if __name__ == '__main__':
    main()
