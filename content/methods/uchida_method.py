import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet

from utils.metrics import *
from utils.logs import *

from content.watermark.secret_key import get_random_watermark
from content.watermark.secret_key import get_watermark_from_text
from content.watermark.secret_key import get_text_from_watermark
from content.watermark.CustomLoss import WatermarkCrossEntropyLoss



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
        #    momentum=self.optimizer_params[1], 
        #    nesterov=True
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
            loss = torch.add(loss, loss_r, alpha=self.las[-1])

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
            loss = torch.add(loss, loss_r, alpha=self.las[-1])
            
        acc = get_accuracy(out, labels)           
        return loss, acc
    
    def _train_one_epoch(self, trainset, validset, verbose=False):
        start_time = datetime.now()
        
        training = []
        for i, batch in enumerate(trainset):
            loss = self._train_one_batch(batch=batch)
            
            training.append(float(loss))
            
            if verbose:
                # if int((10*i) / len(trainset)) % 10 == percent: 
                if (i % 500 == 0) or (i+1 == len(trainset)):
                    ber = 0
                    if self.watermarked:
                        ber = self.get_BER()
                    get_step_logs(i, len(trainset), loss, ber)
            
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
                ber = 0
                if self.watermarked:
                    ber = self.get_BER()    
                get_epoch_logs(
                    epoch, num_epoch, train_loss, val_loss, val_acc, time, ber
                )
                                
        print('model trained') 
        return history
    
    def eval(self, testset, verbose=False):
        acc = get_model_accuracy(self.model, testset)
        acc_per_classes = get_model_accuracy_per_classes(self.model, testset)
        if verbose:
            get_eval_logs(acc, acc_per_classes)
                
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
    
    def prune(self, pruning_ratio, verbose=False):
        parameters_to_prune = (
            (self.model.conv1, 'conv1.weight'),
            (self.model.conv2, 'conv2.weight'),
            (self.model.fc1, 'fc1.weight'),
            (self.model.fc2, 'fc2.weight'),
            (self.model.fc3, 'fc3.weight'),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        layer_sparsity = [get_sparsity(l[0]) for l in parameters_to_prune]
        global_sparsity = get_global_sparsity(parameters_to_prune)
        if verbose:
            get_prune_logs(layer_sparsity, global_sparsity, parameters_to_prune)
        
        return (layer_sparsity, global_sparsity)
    
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
        self._add_watermark(secret_key, method, la)
        
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('watermark rewritted')
        return history
    
    def get_watermark(self, number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        return get_watermark(weights, self.criterion_rs[number].X)
    
    def check_watermark(self, number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        return check_watermark(weights, self.criterion_rs[number].X)
    
    def get_BER(self, number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.model.conv1.weight.detach().to(torch.float32)
        return get_BER(
            weights, self.criterion_rs[number].X, self.secret_keys[number].cpu()
        )

    


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
        to_watermark=True, secret_key='Ettore Hidoux', method="rand", la=5,
        device='mps'
    )
    
    # print(len(net.secret_keys[-1]), net.conv1_size, net.criterion_rs[-1].X.shape)
    # print(net.methods[-1])
    
    # print(net.secret_keys[0].cpu())
    # print(net.criterion_rs[0].X.cpu())
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=32, num_workers=4, pin_memory=True)
    trainset, validset = dataset.get_train_valid_loader()
    
    values = net.train((trainset, validset), num_epoch=3, verbose=2)
    # print(values)
    
    print(net.check_watermark())
    # print(net.get_BER())
    
    #net.load_model('models/mlp_cifar10_20230221_104629')
    
    