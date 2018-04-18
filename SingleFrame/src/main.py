#!/usr/bin/env python3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import get_loaders
from train import train_network
import model

def plot_data(losses, accuracies, name):
    """Plot training and validation statistics.
    Args:
        losses (dictionary): containing list of cross entrophy losses for
                                training and validation splits
        accuracies (dictionary): contains list of accuracies for training
                                    and validation splits
        name (string): name to save plot
    """
    # convert accuracies to percentages
    accuracies['Train'] = [acc * 100 for acc in accuracies['Train']]
    accuracies['Valid'] = [acc * 100 for acc in accuracies['Valid']]
    # set fontsize
    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,5))
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_ylim(0,2)
    ax1.plot(losses['Train'], label='Training')
    ax1.plot(losses['Valid'], label='Validation')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0,100)
    ax2.plot(accuracies['Train'], label='Training')
    ax2.plot(accuracies['Valid'], label='Validation')
    ax2.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('../outputs/' + name)

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
    # plot
    plot_data(losses, accuracies, 'plots.png')
    # save network
    torch.save(net.state_dict(), '../outputs/net_params.pkl')

if __name__ == '__main__':
    main()
