import torch 
import torch.nn as nn
import numpy as np

from .newtorks.Net import Net
from .newtorks.WideResNet import WideResNet

from watermark.secret_key import get_random_watermark, get_watermark_from_text
from watermark.custom_loss import watermarkCrossEntropyLoss

    
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class Network():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(
        self, model_name, model_params, 
        optimizer_params, 
        to_watermark=False, secret_key=None, method="direct", la=10, device='cpu'
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        
        
        self.device = device
        self.model_list = {
            'mlp': Net,
            'wideResNet': WideResNet
        }
        
        self.model = self.model_list[self.model_name](*model_params).to(device)
        self.criterion_0 = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.optimizer_params[0], momentum=self.optimizer_params[0], nesterov=True)
        
        self.to_watermark = to_watermark
        if to_watermark:
            self._watermark_init(secret_key, method, la)
        pass
    
    def _wartermark_init(self, secret_key, method, la):
        self.method = method
        self.la = la
        self.secret_key = secret_key
        if secret_key is None:
            self.secret_key = get_random_watermark(100)
        elif isinstance(secret_key, str):
            self.secret_key = get_watermark_from_text(secret_key)
        
        self.secret_key_size = len(self.secret_key)
        self.conv1_size = len(np.mean(self.model.model.conv1.weight.detach().cpu().numpy(), 0).flatten())
        
        self.secret_key = torch.tensor(self.secret_key, dtype=torch.float32)
        self.criterion_r = watermarkCrossEntropyLoss(type=method, size=(self.secret_key_size, self.conv1_size))
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
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.cpu().to(torch.float32)
            loss += self.la*self.criterion_r(self.secret_key, weights) 
        return loss
    
    def _validation_one_batch(self, batch):
        images, labels = batch 
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        if self.to_watermark:
            weights = self.model.conv1.weight.cpu().to(torch.float32)
            loss += self.la*self.criterion_r(self.secret_key, weights) 
        acc = accuracy(out, labels)           
        return loss, acc
        
    def train(self, set, epoch):
        trainset, validset = set
        # TO DO
        return None
    
    
    def eval(self, testset):
        # TO DO
        return None

    


if __name__ == "__main__":
    import sys 
    
    model_name = sys.argv[1]
    params = sys.argv[2]
    params = params.replace('[', '').replace(']', '').split(',')
    params = [int(param) for param in params]
    print(model_name, params)
    
    net = Network(model_name=model_name, params=params, device='mps')
    net.load_model('models/cifar_net_20230216_003445')
    
    