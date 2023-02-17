import torch

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np



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


def trainset_loader(batch_size: int):
    """
    Args:
        batch_size (int): size of every batch

    Returns:
        _type_: the dataset to train the model on 
    """
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader

def testset_loader(batch_size: int):
    """
    Args:
        batch_size (int): size if every batch

    Returns:
        _type_: the dataset to eval the model on
    """
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return testloader



if __name__ == '__main__':
    
    batch_size = 4

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    trainloader = trainset_loader(batch_size)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))