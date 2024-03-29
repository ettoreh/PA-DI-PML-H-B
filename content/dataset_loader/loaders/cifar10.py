import torch

import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms



# list of classes' names in the cifar dataset
classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 
    'truck'
)

# define the transformation to apply to each image before the network
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def trainset_loader(
    batch_size: int, val_ratio: int, num_workers: int, pin_memory: bool
    ):
    """_summary_

    Args:
        batch_size (int): size if every batch
        val_ration (float): ratio of training vs validation set 
        num_workers (int): number of worker on the task
        pin_memory (bool): activate pin memory

    Returns:
        _type_: the dataset to eval the model on
    """
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    size = len(trainset)
    train_size = int((1-val_ratio)*size)
    train_ds, val_ds = random_split(trainset, [train_size, size-train_size])
    
    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=pin_memory
    )
    validloader = DataLoader(
        val_ds, batch_size=batch_size*2, num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return trainloader, validloader

def testset_loader(batch_size: int, num_workers: int, pin_memory: bool):
    """_summary_

    Args:
        batch_size (int): size if every batch
        num_workers (int): number of worker on the task
        pin_memory (bool): activate pin memory

    Returns:
        _type_: the dataset to eval the model on
    """
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size*2, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    return testloader



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    
    batch_size = 4

    # functions to show an image
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    trainloader, validloader = trainset_loader(batch_size, 0.05, 2, False)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))