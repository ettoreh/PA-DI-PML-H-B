import torch 
import torch.nn as nn
import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet

from content.watermark.secret_key import get_random_watermark
from content.watermark.secret_key import get_watermark_from_text
from content.watermark.custom_loss import watermarkCrossEntropyLoss

    
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



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
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.optimizer_params[0], 
            momentum=self.optimizer_params[0], 
            nesterov=True
        )
        self.to_watermark = to_watermark
        if to_watermark:
            self._watermark_init(secret_key, method, la)
        pass
    
    def _watermark_init(self, secret_key, method, la):
        self.method = method
        self.la = la
        self.secret_key = secret_key
        if secret_key is None:
            self.secret_key = get_random_watermark(100)
        elif isinstance(secret_key, str):
            self.secret_key = get_watermark_from_text(secret_key)
        
        self.secret_key_size = len(self.secret_key)
        self.conv1_size = len(
            np.mean(
                self.model.conv1.weight.detach().cpu().numpy(), 0).flatten()
        )
        self.secret_key = torch.tensor(
            self.secret_key, 
            dtype=torch.float32, 
            device=self.device
        )
        self.criterion_r = watermarkCrossEntropyLoss(
            type=method, 
            size=(self.secret_key_size, self.conv1_size), 
            device=self.device
        )
        print('watermark initiated')
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print('model saved')
    
    def load_matrix(self, matrix_path):
        self.criterion_r.load(matrix_path)
        print('matrix loaded')
        
    def save_matrix(self, matrix_path):
        self.criterion_r.save(matrix_path)
        print('matrix saved')
        
    def _train_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.detach().to(torch.float32)
            loss += self.la*self.criterion_r(weights, self.secret_key) 
        return loss
    
    def _validation_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.detach().to(torch.float32)
            loss += self.la*self.criterion_r(weights, self.secret_key) 
        acc = accuracy(out, labels)           
        return loss, acc
    
    def _train_one_epoch(self, trainset, valideset, verbose=False):
        start_time = datetime.now()
        
        for i, batch in enumerate(trainset):
            loss = self._train_one_batch(batch=batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if verbose:
                # if int((10*i) / len(trainset)) % 10 == percent: 
                if (i % 1000 == 0) or (i+1 == len(trainset)):
                    print(" - Step [{}/{}] \t Loss: {:.4f}"
                        .format(i+1, len(trainset), loss))
        
        validation = [[], []]
        for i, batch in enumerate(valideset):    
            loss, acc = self._validation_one_batch(batch)
            validation[0].append(float(loss))
            validation[1].append(float(acc))
        
        time = datetime.now() - start_time
        return np.mean(validation[0]), np.mean(validation[1]), time
        
    def train(self, set, epoch, verbose=0):
        trainset, validset = set
        # TO DO
        print('Epoch [{}/{}], '.format(1, 3))
        return None
    
    
    def eval(self, testset, verbose=False):
        # TO DO
        return None

    


if __name__ == "__main__":
    import sys 
    from content.dataset_loader.datasetLoader import DatasetLoader
    
    model_name = 'net'
    params = [10]
    print(model_name, params)
    
    net = Network(
        model_name=model_name, model_params=params, to_watermark=True,
        secret_key='Ettore Hidoux', device='mps'
    )
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=8)
    trainset, validset = dataset.get_train_valid_loader()
    
    values = net._train_one_epoch(trainset, validset, True)
    print(values)
    #net.load_model('models/mlp_cifar10_20230221_104629')
    
    