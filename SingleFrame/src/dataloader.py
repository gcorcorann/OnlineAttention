#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
import numbers
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

class ImageDataset(Dataset):
    """Image Attention Level Dataset.

    Args:
        labels_path (string):   path to text file with annotations
        sequence_len (int):     length of video to sample from (starting from
                                    end frame)
        transform (callable):   transform to be applied on image
        
    Returns:
        torch.utils.data.Dataset:   dataset object
    """
    def __init__(self, labels_path, sequence_len, transform=None):
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))
        
        self.data = data
        self.sequence_len = sequence_len
        self.transform = transform
        self.cap = cv2.VideoCapture()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = int(self.data[idx, 1]) - 1
        video_path = self.data[idx, 0]
        # open video
        self.cap.open(video_path)
        # random index (counting backwards from last frame)
        i = np.random.randint(self.sequence_len) + (100-self.sequence_len)
        # set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, X = self.cap.read()
        X = cv2.resize(X, (256,256))
        # convert from BGR to RGB
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        # transform data
        if self.transform:
            X = self.transform(X)
        # store in sample
        sample = {'X': X, 'y': y}
        return sample

class RandomRotation():
    """Rotation image by an angle.

    Args:
        degress (float or int):     range of degrees to rotation image.
    """
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, image):
        """
        Args:
            image (PIL Image:       image to be rotated

        Returns:
            PIL image:              rotated image
        """
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        return image.rotate(angle)

def count_labels(data_path):
    """Count the number of instances in each class.

    Args:
        data_path (string):     Path to annotations.

    Returns:
        counts (ndarray):       array containing number of instances.
    """
    counts = np.zeros(4, dtype=int)
    with open(data_path, 'r') as f:
        for line in f:
            line = int(line.split()[1]) - 1
            counts[line] += 1

    return counts

def get_loaders(train_path, valid_path, batch_size, sequence_len,
        num_workers, gpu):
    """Return dictionary of torch.utils.data.DataLoader.

    Args:
        train_path (string):    path to training annotations
        valid_path (string):    path to validation annotations
        batch_size (int):       number of instances in batch
        sequence_len (int):     number of frames to keep in video (counting
                                    from last frame)
        num_workers (int):      number of subprocesses used for data loading
        gpu (bool):             presence of gpu

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        dictionary:                     dataset length for training and 
                                            validation
    """
    # data augmentation and normalization for training
    # just normalization for validation
    data_transforms = {
            'Train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224,0.225])
                ]),
            'Valid': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224,0.225])
                ])
            }
    # create dataset object
    datasets = {
            'Train': ImageDataset(train_path, sequence_len,
                data_transforms['Train']),
            'Valid': ImageDataset(valid_path, sequence_len,
                data_transforms['Valid'])
            }

    # dataset sizes
    dataset_sizes = {'Train': len(datasets['Train']),
                     'Valid': len(datasets['Valid'])}

    # add weighted sampler since unbalanced dataset
    c = count_labels(train_path)
    weights = np.zeros(dataset_sizes['Train'])
    weights[: c[0]] = c[1] / c[0]
    weights[c[0]: c[0]+c[1]] = 1
    weights[c[0]+c[1]:c[0]+c[1]+c[2]] = c[1] / c[2]
    weights[c[0]+c[1]+c[2]:] = c[1] / c[3]

    # create dataloders
    dataloaders = {
            'Train': DataLoader(datasets['Train'], batch_size=batch_size,
                sampler=WeightedRandomSampler(weights, dataset_sizes['Train'])),
            'Valid': DataLoader(datasets['Valid'], batch_size=batch_size,
                shuffle=True)
            }

    return dataloaders, dataset_sizes

def main():
    """Main Function."""
    import matplotlib.pyplot as plt
    from torchvision import utils

    # data paths
    train_path = '/home/gary/datasets/accv/labels/train_data.txt'
    valid_path = '/home/gary/datasets/accv/labels/valid_data.txt'
    test_path = '/home/gary/datasets/accv/labels/test_data.txt'
    # hyper-parameters
    batch_size = 24
    sequence_len = 50  # counting backwards from last frame

    dataloaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, sequence_len, 0, False)
    print('Dataset Sizes:')
    print(dataset_sizes)

    def imshow(grid):
        grid = grid.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        grid = std * grid + mean
        grid = np.clip(grid, 0, 1)
        plt.imshow(grid)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

    train_batch = next(iter(dataloaders['Train']))
    data, labels = train_batch['X'], train_batch['y']
    print('data:', data.shape)
    print('labels:', labels)
    grid = utils.make_grid(data)
    imshow(grid)

if __name__ == '__main__':
    main()
