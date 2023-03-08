import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet
from networks.Perceptron import SinglePerceptron

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
        layers_to_watermark, secret_key, method, l1, l2,
        device
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
            
        self._init_watermark()
        if len(layers_to_watermark) > 0:
            self._add_watermark(secret_key, layers_to_watermark, method, l1, l2)
        pass
    
    def _init_watermark(self):
        self.to_watermark = False
        self.methods = []
        self.l1s = []
        self.l2s = []
        self.secret_keys = []
        self.layer_sizes = {}
        for name, layer in self.model.layers.items():
            self.layer_sizes[name] = len(
                np.mean(layer.weight.detach().cpu().numpy(), 0).flatten()
            )
        self.criterion_rs = []
        self.perceptrons = []
        self.optimizer_ps = []
        self.layers_to_watermark = []
        print('watermark initiated')
        
    def _add_watermark(self, secret_key, layers_to_watermark, method, l1, l2):
        self.to_watermark = True
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
        
        
        criterion_r_per_layer = {}
        perceptron_per_layer = {}
        optimizer_p = {}
        self.layers_to_watermark.append(layers_to_watermark)
        for name in self.layers_to_watermark[-1]:
            criterion_r_per_layer[name] = WatermarkCrossEntropyLoss(
                type=method, 
                size=(len(secret_key), self.layer_sizes[name]), 
                device=self.device
            )
            perceptron_per_layer[name] = self.model_list[
                'perceptron'](*[self.layer_sizes[name], 1]).to(self.device
            )
            optimizer_p[name] = torch.optim.SGD(
                self.perceptrons[-1].parameters(), lr=0.01
            )
            
        self.criterion_rs.append(criterion_r_per_layer)
        self.perceptrons.append(perceptron_per_layer)
        self.optimizer_ps.append(optimizer_p)
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
        
    # TO DO
    def _train_one_batch(self, batch):
        return None