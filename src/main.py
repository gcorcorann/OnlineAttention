#!/usr/bin/env python3
import torch
from dataloader import get_loaders

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    train_path = '/home/gary/datasets/accv/labels/train_data.txt'
    valid_path = '/home/gary/datasets/accv/labels/valid_data.txt'
    batch_size = 10
    sequence_len = 50
    num_workers = 1
    
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path,
            batch_size, sequence_len, num_workers, gpu)
    print(dataset_sizes)

    

if __name__ == '__main__':
    main()
