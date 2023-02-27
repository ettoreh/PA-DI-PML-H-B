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
        optimizer, optimizer_params, 
        layers_to_watermark, secret_key, method, la,
        device
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
        
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optimizer_params[0],
                momentum=self.optimizer_params[1], 
                nesterov=True
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.optimizer_params[0]
            )
            
        self.layers_to_watermark = layers_to_watermark
        self._init_watermark()
        if len(layers_to_watermark) > 0:
            self._add_watermark(secret_key, method, la)
        pass
    
    def _init_watermark(self):
        self.to_watermark = False
        self.methods = []
        self.las = []
        self.secret_keys = []
        self.layer_sizes = {}
        for name, layer in self.model.layers.items():
            self.layer_sizes[name] = len(
                np.mean(layer.weight.detach().cpu().numpy(), 0).flatten()
            )
        self.criterion_rs = []
        print('watermark initiated')
        
    def _add_watermark(self, secret_key, method, la):
        self.to_watermark = True
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
        criterion_r_per_layer = {}
        for name in self.layers_to_watermark:
            criterion_r_per_layer[name] = WatermarkCrossEntropyLoss(
                type=method, 
                size=(len(secret_key), self.layer_sizes[name]), 
                device=self.device
            )
        self.criterion_rs.append(criterion_r_per_layer)
        print('watermark parameters added')
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print('model saved')
    
    def load_matrix(self, matrix_path, watermark_number=-1, layer_name='conv1'):
        self.criterion_rs[watermark_number][layer_name].load(matrix_path)
        print('matrix loaded')
        
    def save_matrix(self, matrix_path, watermark_number=0, layer_name='conv1'):
        self.criterion_rs[watermark_number][layer_name].save(matrix_path)
        print('matrix saved')
        
    def _train_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            self.watermarked = True
            for name in self.layers_to_watermark:
                weights = self.get_weights(name)
                loss_r = self.criterion_rs[-1][name](weights, self.secret_keys[-1]) 
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
            for name in self.layers_to_watermark:
                weights = self.get_weights(name)
                loss_r = self.criterion_rs[-1][name](weights, self.secret_keys[-1]) 
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
                    ber = []
                    if self.watermarked:
                        ber = [self.get_BER(name) for name in self.layers_to_watermark]
                    get_step_logs(i, len(trainset), loss, ber)
            
        validation = [[], []]
        for i, batch in enumerate(validset):    
            loss, acc = self._validation_one_batch(batch)
            validation[0].append(float(loss))
            validation[1].append(float(acc))
        
        time = datetime.now() - start_time
        return (
            np.mean(training), np.mean(validation[0]), np.mean(validation[1]),
            str(time).split(".")[0]
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
                ber = []
                if self.watermarked:
                    ber = [self.get_BER(name) for name in self.layers_to_watermark]  
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
        self, set, num_epoch, layers_to_watermark=[], secret_key=None, 
        method='direct', la=10, verbose=0
    ):
        if len(layers_to_watermark) > 0:
            self.layers_to_watermark = layers_to_watermark
            if not self.watermarked:
                self._init_watermark()
            self._add_watermark(secret_key, method, la)
        else:
            self.to_watermark = False
            
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('model fine-tuned')
        return history
    
    def prune(self, pruning_ratio, global_=True, verbose=False):
        
        self.model.to('cpu')
        parameters_to_prune = self.model.prune_layers
        
        if global_:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
        else:
            for module, name in parameters_to_prune:
                prune.l1_unstructured(module, name=name, amount=pruning_ratio)
        
        for module, name in parameters_to_prune:        
            prune.remove(module, name='weight')
        
        layer_sparsity = [get_sparsity(l[0].weight) for l in parameters_to_prune]
        global_sparsity = get_global_sparsity(parameters_to_prune)
        if verbose:
            get_prune_logs(layer_sparsity, global_sparsity, self.model.layers)
        
        self.model.to(self.device)
        
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
            print('key: ', secret_key)
        self.to_watermark = True 
        self.layers_to_watermark['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
        self._add_watermark(secret_key, method, la)
        
        history = self.train(set=set, num_epoch=num_epoch, verbose=verbose)
        print('watermark rewritted')
        return history
    
    def get_weights(self, name, detach=False):
        if detach:
            return self.model.layers[name].weight.detach().to(torch.float32)
        return self.model.layers[name].weight.to(torch.float32)
    
    def get_watermark(self, layer_name, matrix_number=-1,):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.get_weights(layer_name, detach=True)
        return get_watermark(weights, self.criterion_rs[matrix_number][layer_name].X)
    
    def check_watermark(self, layer_name, matrix_number=-1,):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.get_weights(layer_name, detach=True)
        return check_watermark(weights, self.criterion_rs[matrix_number][layer_name].X)
    
    def get_BER(self, layer_name, matrix_number=-1, key_number=0):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        
        weights = self.get_weights(layer_name, detach=True)
        return get_BER(
            weights, 
            self.criterion_rs[matrix_number][layer_name].X, 
            self.secret_keys[key_number].cpu()
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
        optimizer = 'adam', optimizer_params=[0.001, 0.9], 
        layers_to_watermark=['conv1', 'conv2'], secret_key='Ettore Hidoux', method="rand", la=5,
        device='mps'
    )
    
    for k, v in net.model.layers.items():
        print(v)
        
    # print(net.model.layers)
    # print(net.model.prune_layers)
    
    # print(len(net.secret_keys[-1]), net.conv1_size, net.criterion_rs[-1].X.shape)
    # print(net.methods[-1])
    
    # print(net.secret_keys[0].cpu())
    # print(net.criterion_rs[0].X.cpu())
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=32, num_workers=4, pin_memory=True)
    trainset, validset = dataset.get_train_valid_loader()
    
    values = net.train((trainset, validset), num_epoch=1, verbose=2)
    # print(values)

    for name in net.layers_to_watermark:
        print(name, net.check_watermark(name))
    # print(net.get_BER())
    
    # values = net.fine_tune((trainset, validset), 2, verbose=2)
    # for name in net.layers_to_watermark:
    #     print(name, net.check_watermark(name))
    
    value = net.prune(0.3, verbose=True)
    for name in net.layers_to_watermark:
        print(name, net.check_watermark(name))
        
    value = net.rewrite((trainset, validset), 3, 'New Watermark', 'rand', 10, 2)
    for name in net.layers_to_watermark:
        print(name, net.check_watermark(name))
    #net.load_model('models/mlp_cifar10_20230221_104629')
    
    