import torch
import torch.nn as nn

from datetime import datetime

from Net import Net
from load_dataset import testloader, classes



def get_model_accuracy(net, testloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device) 
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct // total

def get_model_accuracy_per_classes(net, testloader, classes, device):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)    
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    return correct_pred, total_pred

    
    
if __name__ == '__main__':
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('on machine ', device)
    model = Net(10).to(device)
    model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models/cifar_net_20230216_003445'
    model.load_state_dict(torch.load(model_path))
    print('model loaded')
    
    accuracy = get_model_accuracy(model, testloader, device)
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
    
    # print accuracy for each class
    correct_pred, total_pred = get_model_accuracy_per_classes(model, testloader, classes, device)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    