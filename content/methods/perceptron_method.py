import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet
from networks.Perceptron import SinglePerceptron

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
    
def _get_sparsity(layer):
        return 100. * float(torch.sum(layer == 0)) / float(layer.nelement())
    
def _get_global_sparsity(layers):
    return 100. * float(
        torch.sum(layers[0][0].weight == 0)
        + torch.sum(layers[1][0].weight == 0)
        + torch.sum(layers[2][0].weight == 0)
        + torch.sum(layers[3][0].weight == 0)
        + torch.sum(layers[4][0].weight == 0)
    ) / float(
        layers[0][0].weight.nelement()
        + layers[1][0].weight.nelement()
        + layers[2][0].weight.nelement()
        + layers[3][0].weight.nelement()
        + layers[4][0].weight.nelement()
    )



class Network():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(
        self, model_name, model_params, 
        optimizer_params=[0.01, 0.9], 
        to_watermark=False, secret_key=None, method="direct", l1=10, l2=10,
        device='cpu'
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        
        self.device = device
        self.model_list = {
            'net': Net,
            'wideResNet': WideResNet,
            'perceptron': SinglePerceptron,
        }
        
        self.model = self.model_list[self.model_name](*model_params).to(device)
        self.criterion_0 = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.optimizer_params[0], 
        #    momentum=self.optimizer_params[1], 
        #    nesterov=True
        )
        self.to_watermark = to_watermark
        if to_watermark:
            self._init_watermark()
            self._add_watermark(secret_key, method, l1, l2)
        pass
    
    def _init_watermark(self):
        self.methods = []
        self.l1s = []
        self.l2s = []
        self.secret_keys = []
        self.conv1_size = len(
            np.mean(
                self.model.conv1.weight.detach().cpu().numpy(), 0).flatten()
        )
        self.criterion_rs = []
        self.perceptrons = []
        self.optimizer_ps = []
        print('watermark initiated')
        
    def _add_watermark(self, secret_key, method, l1, l2):
        self.methods.append(method)
        self.l1s.append(l1)
        self.l2s.append(l2)
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
        self.perceptrons.append(
            self.model_list['perceptron'](*[self.conv1_size, 2, 1]).to(self.device)
        )
        self.optimizer_ps.append(
            torch.optim.SGD(self.perceptrons[-1].parameters(), lr=0.01)
        )
        
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
            
            out_p = self.perceptrons[-1](self.criterion_rs[-1].X)
            loss_p = self.criterion_0(out_p.flatten(), self.secret_keys[-1])
            
            weights_p = self.perceptrons[-1].linear1.weight.to(torch.float32)
            dist_p = torch.mean((torch.mean(weights, 0).flatten() - weights_p)**2)
            
            loss = torch.add(loss, loss_p, alpha=self.l1s[-1])
            loss = torch.add(loss, dist_p, alpha=self.l2s[-1])

        # Backward and optimize
        self.optimizer.zero_grad()
        self.optimizer_ps[-1].zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_ps[-1].step()
        
        return loss
    
    
    def _validation_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.to(torch.float32)
            
            out_p = self.perceptrons[-1](self.criterion_rs[-1].X)
            loss_p = self.criterion_0(out_p.flatten(), self.secret_keys[-1])
            
            weights_p = self.perceptrons[-1].linear1.weight.to(torch.float32)
            dist_p = torch.mean((torch.mean(weights, 0).flatten() - weights_p)**2)
            
            loss = torch.add(loss, loss_p, alpha=self.l1s[-1])
            loss = torch.add(loss, dist_p, alpha=self.l2s[-1])
            
        acc = accuracy(out, labels)           
        return loss, acc
    
    # Same, To move
    def _train_one_epoch(self, trainset, validset, verbose=False):
        start_time = datetime.now()
        
        training = []
        for i, batch in enumerate(trainset):
            loss = self._train_one_batch(batch=batch)
            
            training.append(float(loss))
            
            if verbose:
                # if int((10*i) / len(trainset)) % 10 == percent: 
                if (i % 200 == 0) or (i+1 == len(trainset)):
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
        
    # Same, To move
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
                                
        print('model trained') 
        return history
    
    # Same, To move
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
    
    # Same, To move
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
    
    # Same, To move
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
    
    # Almost same, To move
    def fine_tune(
        self, set, num_epoch, with_watermark=False, secret_key=None, 
        method='direct', l1=10, l2=5, verbose=0
    ):
        if with_watermark:
            if not self.watermarked:
                self.to_watermark = True
                self._init_watermark()
            self._add_watermark(secret_key, method, l1, l2)
        else:
            self.to_watermark = False
            
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('model fine-tuned')
        return history
    
    # Same, To move
    def prune(self, pruning_ratio, verbose=False):
        parameters_to_prune = (
            (self.model.conv1, 'weight'),
            (self.model.conv2, 'weight'),
            (self.model.fc1, 'weight'),
            (self.model.fc2, 'weight'),
            (self.model.fc3, 'weight'),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        conv1_sparsity = _get_sparsity(parameters_to_prune[0][0].weight)
        conv2_sparsity = _get_sparsity(parameters_to_prune[1][0].weight)
        fc1_sparsity = _get_sparsity(parameters_to_prune[2][0].weight)
        fc2_sparsity = _get_sparsity(parameters_to_prune[3][0].weight)
        fc3_sparsity = _get_sparsity(parameters_to_prune[4][0].weight)
        global_sparsity = _get_global_sparsity(parameters_to_prune)
        if verbose:
            print("Sparsity in conv1.weight: {:.2f}%".format(conv1_sparsity))
            print("Sparsity in conv2.weight: {:.2f}%".format(conv2_sparsity))
            print("Sparsity in fc1.weight: {:.2f}%".format(fc1_sparsity))
            print("Sparsity in fc2.weight: {:.2f}%".format(fc2_sparsity))
            print("Sparsity in fc3.weight: {:.2f}%".format(fc3_sparsity))
            print("Global sparsity: {:.2f}%".format(global_sparsity))
        
        return (conv1_sparsity, conv2_sparsity, fc1_sparsity, fc2_sparsity, 
                fc3_sparsity, global_sparsity)
    
    # Almost same, To move
    def rewrite(
        self, set, num_epoch, secret_key=None, method='direct', l1=10, l2=5,
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
        self._add_watermark(secret_key, method, l1, l2)
        
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('watermark rewritted')
        return history
        
    # Same, To move
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
    
    # Same, To move
    def get_BER(self, watermark_number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        weights = torch.mean(weights, 0)
        weights = torch.matmul(self.criterion_rs[-1].X, weights.flatten())
        weights = self.criterion_rs[-1].sigmoid(weights)
        weights = (weights.detach().cpu().numpy() > 0.5).astype(int)
        return np.mean([weights[i] != self.secret_keys[watermark_number].cpu()[i] for i in range(len(weights))]) 

    


if __name__ == "__main__":
    import sys 
    from content.dataset_loader.datasetLoader import DatasetLoader
    
    model_name = 'net'
    params = [10, [6, 16], 5]
    # print(model_name, params)
    
    # secret_key='Copyright from Ettore Hidoux',
    net = Network(
        model_name=model_name, model_params=params, 
        optimizer_params=[0.001, 0.9], 
        to_watermark=True, secret_key='Ettore Hidoux', method="rand", l1=1, l2=10
        
        ,
        device='mps'
    )
    
    print(len(net.secret_keys[-1]), net.conv1_size, net.criterion_rs[-1].X.shape)
    # print(net.criterion_p)
    print(net.perceptrons[-1].linear1.weight.shape)
    # print(net.methods[-1])
    
    # print(net.secret_keys[0].cpu())
    # print(net.criterion_rs[0].X.cpu())
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=32, num_workers=4, pin_memory=True)
    trainset, validset = dataset.get_train_valid_loader()
    
    # values = net._train_one_epoch(trainset, validset, verbose=True)
    # print(values)
    
    values = net.train((trainset, validset), num_epoch=3, verbose=2)
    
    print(net.check_watermark())
    