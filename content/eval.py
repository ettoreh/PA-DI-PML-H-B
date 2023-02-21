import torch
import torch.nn as nn



def get_model_accuracy(net: nn.Module, data) -> float:
    """_summary_

    Args:
        net (nn.Module): the neural network used for predictions
        data (_type_): the data on which the network is tested 

    Returns:
        float: the overall accuracy percentage 
    """
    correct, total = 0, 0
    # no training, no gradients needed 
    with torch.no_grad():
        for images, labels in data:
            images, labels = images, labels
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is choosed as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct // total

def get_model_accuracy_per_classes(net: nn.Module, data, classes: list) -> dict:
    """_summary_

    Args:
        net (nn.Module): the neural network used for predictions
        data (_type_): the data on which the network is tested
        classes (list): the list of classe names

    Returns:
        dict: the accuracy percentage per classe
    """
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    percent_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        for images, labels in data:
            images, labels = images, labels  
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        percent_pred[classname] = accuracy
        
    return percent_pred

    
    
if __name__ == '__main__':
    import sys
    
    from dataset_loader.datasetLoader import DatasetLoader
    from networks.networks import Network
    


    model = sys.argv[1]
    data = sys.argv[2]
    params = sys.argv[3]
    params = params.replace('[', '').replace(']', '').split(',')
    params = [int(param) for param in params]
    model_path = sys.argv[4]
    print(model)
    print(data)
    print(params)
    print(model_path)
    
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # print('on machine ', device)
    
    # model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models/cifar_net_20230216_003445'
    model = Network(model_name=model, params=params)
    model.load_model(model_path=model_path)
    
    dataset = DatasetLoader(dataset_name=data, batch_size=4)
    test_data = dataset.testset_loader()
    
    accuracy = get_model_accuracy(model.model, test_data)
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
    
    percent_pred = get_model_accuracy_per_classes(model.model, test_data, dataset.classes)
    for classname, acc in percent_pred.items():
        print(f'Accuracy for class: {classname:5s} is {acc:.1f} %')