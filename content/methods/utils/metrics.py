import torch 
import numpy as np

from content.watermark.secret_key import get_text_from_watermark



def get_watermark(weights, X):
    weights = torch.mean(weights, 0)
    weights = torch.matmul(X, weights.flatten())
    weights = torch.sigmoid(weights)
    weights = (weights.detach().cpu().numpy() > 0.5).astype(int)
    return weights

def check_watermark(weights, X):
    weights = get_watermark(weights, X)
    return get_text_from_watermark(weights)
    
def get_BER(weights, X, secret_key):
    weights = get_watermark(weights, X)
    return np.mean([weights[i] != secret_key[i] for i in range(len(weights))]) 

def get_model_accuracy(model, data):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct) / total
    
def get_model_accuracy_per_classes(model, data, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    percent_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for images, labels in data:
            images, labels = images, labels  
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        percent_pred[classname] = accuracy

    return percent_pred

def get_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_sparsity(layer):
    return 100. * float(torch.sum(layer == 0)) / float(layer.nelement())
    
def get_global_sparsity(layers):
    value = 100. * float(sum([torch.sum(l[0].weight == 0) for l in layers]))
    return value / float(sum([l[0].weight.nelement() for l in layers]))
