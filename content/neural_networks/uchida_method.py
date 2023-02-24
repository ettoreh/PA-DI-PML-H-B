import torch 
import torch.nn as nn
import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet

from content.watermark.secret_key import get_random_watermark
from content.watermark.secret_key import get_watermark_from_text
from content.watermark.secret_key import get_text_from_watermark
from content.watermark.CustomLoss import WatermarkCrossEntropyLoss

    
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def _generate_logs(epoch, num_epoch, train_loss, val_loss, val_acc, time):
        trainlog = "Epoch [{}/{}], \t ".format(epoch+1, num_epoch)
        trainlog += "Train loss: {:.4f}, \t ".format(train_loss)
        trainlog += "Validation loss: {:.4f}, \t ".format(val_loss)
        trainlog += "Validation acc: {:.4f}, \t ".format(val_acc)
        trainlog += "Time: {}".format(time)
        return trainlog    



class Network():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(
        self, model_name, model_params, 
        optimizer_params=[0.01, 0.9], 
        to_watermark=False, secret_key=None, method="direct", la=10,
        device='cpu'
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        
        self.device = device
        self.model_list = {
            'net': Net,
            'wideResNet': WideResNet
        }
        
        self.model = self.model_list[self.model_name](*model_params).to(device)
        self.criterion_0 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.optimizer_params[0], 
            #momentum=self.optimizer_params[1], 
            #nesterov=True
        )
        self.to_watermark = to_watermark
        if to_watermark:
            self._init_watermark()
            self._add_watermark(secret_key, method, la)
        pass
    
    def _init_watermark(self):
        self.methods = []
        self.las = []
        self.secret_keys = []
        self.conv1_size = len(
            np.mean(
                self.model.conv1.weight.detach().cpu().numpy(), 0).flatten()
        )
        self.criterion_rs = []
        print('watermark initiated')
        
    def _add_watermark(self, secret_key, method, la):
        self.methods.append(method)
        self.las.append(la)
        if secret_key is None:
            secret_key = get_random_watermark(100)
        elif isinstance(secret_key, str):
            secret_key = get_watermark_from_text(secret_key)
        self.secret_keys.append(torch.tensor(
            secret_key, 
            dtype=torch.float32, 
            device=self.device
        ))
        self.criterion_rs.append(WatermarkCrossEntropyLoss(
            type=method, 
            size=(len(secret_key), self.conv1_size), 
            device=self.device
        ))
        print('watermark added')
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print('model saved')
    
    def load_matrix(self, matrix_path, watermark_number=-1):
        self.criterion_rs[watermark_number].load(matrix_path)
        print('matrix loaded')
        
    def save_matrix(self, matrix_path, watermark_number=0):
        self.criterion_rs[watermark_number].save(matrix_path)
        print('matrix saved')
        
    def _train_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            self.watermarked = True
            weights = self.model.conv1.weight.to(torch.float32)
            loss_r = self.criterion_rs[-1](weights, self.secret_keys[-1]) 
            loss = loss + self.las[-1]*loss_r

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def _validation_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.to(torch.float32)
            loss_r = self.criterion_rs[-1](weights, self.secret_keys[-1]) 
            loss = loss + self.las[-1]*loss_r
        acc = accuracy(out, labels)           
        return loss, acc
    
    def _train_one_epoch(self, trainset, validset, verbose=False):
        start_time = datetime.now()
        
        training = []
        for i, batch in enumerate(trainset):
            loss = self._train_one_batch(batch=batch)
            
            training.append(float(loss))
            
            if verbose:
                # if int((10*i) / len(trainset)) % 10 == percent: 
                if (i % 1000 == 0) or (i+1 == len(trainset)):
                    print(" - Step [{}/{}] \t Loss: {:.4f} \t BER: {:.4f}"
                        .format(i+1, len(trainset), loss, self.get_BER()))
        
        validation = [[], []]
        for i, batch in enumerate(validset):    
            loss, acc = self._validation_one_batch(batch)
            validation[0].append(float(loss))
            validation[1].append(float(acc))
        
        time = datetime.now() - start_time
        return (
            np.mean(training), np.mean(validation[0]), np.mean(validation[1]),
            time
        )
        
    def train(self, set, num_epoch, verbose=0):
        print("training started")
        trainset, validset = set
        history = []
        for epoch in range(num_epoch):
            print("Epoch [{}/{}]".format(epoch+1, num_epoch))
            
            self.model.train(True)
            train_loss, val_loss, val_acc, time = self._train_one_epoch(
                trainset=trainset, validset=validset, verbose=(verbose==2)
            )
            
            history.append([train_loss, val_loss, val_acc, time])
            if (verbose > 0):
                print(_generate_logs(
                    epoch, num_epoch, train_loss, val_loss, val_acc, time))
                
                print(self.get_BER())
                
        print('model trained') 
        return history
    
    def _get_model_accuracy(self, data):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in data:
                images, labels = images, labels
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (100 * correct) / total
    
    def _get_model_accuracy_per_classes(self, data, classes):
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        percent_pred = {classname: 0 for classname in classes}
        with torch.no_grad():
            for images, labels in data:
                images, labels = images, labels  
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            percent_pred[classname] = accuracy
            
        return percent_pred
    
    def eval(self, testset, verbose=False):
        acc = self._get_model_accuracy(testset)
        acc_per_classes = self._get_model_accuracy_per_classes(testset)
        if verbose:
            print(f'Accuracy of the network on the 10000 test images: {acc} %')
            print(f'***')
            for classname, acc in acc_per_classes.items():
                print(f'Accuracy for class: {classname:5s} is {acc:.1f} %')
                
        print('model evaluated')
        return acc, acc_per_classes
    
    def fine_tune(
        self, set, num_epoch, with_watermark=False, secret_key=None, 
        method='direct', la=10, verbose=0
    ):
        if with_watermark:
            if not self.watermarked:
                self.to_watermark = True
                self._init_watermark()
            self._add_watermark(secret_key, method, la)
        else:
            self.to_watermark = False
            
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('model fine-tuned')
        return history
    
    def prune(self, verbose=False):
        # TO DO
        return None
    
    def rewrite(
        self, set, num_epoch, secret_key=None, method='direct', la=10,
        verbose=0
    ):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        if secret_key is None:
            secret_key = get_random_watermark(150)
        if (verbose > 0):
            print('key: ', get_text_from_watermark(secret_key))
        self.to_watermark = True 
        self._add_watermark(secret_key, method, la, rewrite=True)
        
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('watermark rewritted')
        return history
    
    def check_watermark(self):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        weights = torch.mean(weights, 0)
        weights = torch.matmul(self.criterion_rs[-1].X, weights.flatten())
        weights = self.criterion_rs[-1].sigmoid(weights)
        weights = (weights.detach().cpu().numpy() > 0.5).astype(int)
        print(weights)
        return get_text_from_watermark(weights)
    
    def get_BER(self, watermark_number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        weights = torch.mean(weights, 0)
        weights = torch.matmul(self.criterion_rs[-1].X, weights.flatten())
        weights = self.criterion_rs[-1].sigmoid(weights)
        weights = (weights.detach().cpu().numpy() > 0.5).astype(int)
        return np.mean([weights[i] == self.secret_keys[watermark_number].cpu()[i] for i in range(len(weights))]) 

    


if __name__ == "__main__":
    import sys 
    from content.dataset_loader.datasetLoader import DatasetLoader
    
    model_name = 'net'
    params = [10, [20, 50], 5]
    print(model_name, params)
    
    net = Network(
        model_name=model_name, model_params=params, 
        optimizer_params=[0.001, 0.9], 
        to_watermark=True, secret_key='Ettore Hidoux', method="rand", la=10,
        device='mps'
    )
    
    print(len(net.secret_keys[-1]), net.conv1_size, net.criterion_rs[-1].X.shape)
    print(net.methods[-1])
    
    print(net.secret_keys[0].cpu())
    print(net.criterion_rs[0].X.cpu())
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=8)
    trainset, validset = dataset.get_train_valid_loader()
    
    values = net.train((trainset, validset), num_epoch=3, verbose=2)
    # print(values)
    
    print(net.check_watermark())
    print(net.get_BER())
    
    #net.load_model('models/mlp_cifar10_20230221_104629')
    
    