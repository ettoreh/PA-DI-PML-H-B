import torch

from WideResNet import WideResNet
from load_dataset import testloader, classes

    
    
if __name__ == '__main__':
    model_path = ''
    model = WideResNet()
    model.load_state_dict(torch.load(model_path))
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print('---')
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')    