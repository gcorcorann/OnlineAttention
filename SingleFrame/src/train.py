#!/usr/bin/env python3
import time
import copy
import torch
from torch.autograd import Variable

def train_network(net, dataloaders, dataset_sizes, batch_size, sequence_len,
        criterion, optimizer, max_epochs, gpu):
    """Train network.

    Args:
        net (torchvision.models):   network to train
        dataloaders (dictionary):   contains torch.utils.data.DataLoader 
                                        for both training and validation
        dataset_sizes (dictionary): size of training and validation datasets
        batch_size (int):           size of mini-batch
        sequence_len (int):         length of video sequence
        criterion (torch.nn.modules.loss):  loss function
        opitimier (torch.optim):    optimization algorithm.
        max_epochs (int):           max number of epochs used for training
        gpu (bool):                 gpu availability
    """
    # start timer
    start = time.time()
    # store network to gpu
    if gpu:
        net = net.cuda()

    # store best validation accuracy and corresponding model
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0
    losses = {'Train': [], 'Valid': []}
    accuracies = {'Train': [], 'Valid': []}
    patience = 0
    for epoch in range(max_epochs):
        print()
        print('Epoch', epoch+1)
        print('-' * 8)
        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                net.train(True)  # set model to training model
            else:
                net.train(False)  # set model to evaluation mode

            # epoch statistics
            running_loss = 0
            running_correct = 0
            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data['X'], data['y']
                print('inputs:', inputs.shape)
                print('labels:', labels.shape)
                # wrap in Variable
                if gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # pass through network
                outputs = net.forward(inputs)
                print('outputs:', outputs.shape)
                # loss + prediction
                loss = criterion(outputs, labels)
                print('loss:', loss)
                _, pred = torch.max(outputs.data, 1)
                correct = torch.sum(pred == labels.data)
                print('correct:', correct)

                # back-prop + optimize in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_correct += correct

            epoch_loss = running_loss * batch_size / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # store stats
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            if phase == 'Valid':
                patience += 1
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # deep copy model
                    best_model_wts = copy.deepcopy(net.state_dict())
                    patience = 0

        if patience == 20:
            break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}h {:.9f}m'.format(
        time_elapsed // (60*60), time_elapsed // 60 % 60
        ))
    # load best model weights
    net.load_state_dict(best_model_wts)
    return network, best_acc, losses, accuracies

