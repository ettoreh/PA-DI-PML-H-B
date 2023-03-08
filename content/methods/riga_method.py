import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np

from datetime import datetime

from networks.Net import Net
from networks.WideResNet import WideResNet
from networks.Perceptron import SinglePerceptron
from networks.GAN import RIGA, Generator, Discriminator, Extractor

from utils.metrics import *
from utils.logs import *

from content.watermark.secret_key import get_random_watermark
from content.watermark.secret_key import get_watermark_from_text
from content.watermark.secret_key import get_text_from_watermark
from content.watermark.secret_matrix import get_secret_matrix



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
            'riga': RIGA,
            'gen': Generator,
            'det': Discriminator,
            'ext': Extractor,
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
    
    # TO DO
    def _init_watermark(self):
        self.to_watermark = False
        self.methods = []
        self.l1s = []
        self.l2s = []
        self.secret_keys = []
        self.layers_to_watermark = []
        self.layer_sizes = {}
        for name, layer in self.model.layers.items():
            self.layer_sizes[name] = len(
                np.mean(layer.weight.detach().cpu().numpy(), 0).flatten()
            )
        self.discriminators = []
        self.extractors = []
        print('watermark initiated')
        
    # TO DO 
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
        
        dicriminators_per_layer = {}
        extractors_per_layer = {}
        self.layers_to_watermark.append(layers_to_watermark)
        for name in self.layers_to_watermark[-1]:
            dicriminators_per_layer[name] = Discriminator(
                self.layer_sizes[name],
                2*self.layer_sizes[name]
            ).to(self.device)
            
            extractors_per_layer[name] = Extractor(
                self.layer_sizes[name],
                2*self.layer_sizes[name],
                len(secret_key)
            ).to(self.device)
            
        self.discriminators.append(dicriminators_per_layer)
        self.extractors.append(extractors_per_layer)
        print('watermark parameters added')
        
    # TO DO
    def _train_one_batch(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        
        if self.to_watermark:
            self.watermarked = True
            
            for name in self.layers_to_watermark[-1]:
                
                weights = self.get_weights(name)
                
                loss_wm = self.get_BER(name, self.extractors[-1][name], self.secret_keys[-1])
                loss = torch.add(loss, loss_wm, alpha=self.l1s[-1])
                
                loss_det = self.discriminators[-1][name](weights)
                loss = torch.add(loss, loss_det, alpha=self.l2s[-1])

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
        
        
        
        
        
        
        
        
        
        
        
    def get_weights(self, detach=False):
        params = []
        for prm in self.model.parameters():
            params.append(torch.mean(prm, 0).flatten())
        if detach:
            return torch.cat(params).detach()
        return torch.cat(params)
    
    def get_watermark(self, extractor):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        weights = self.get_weights(detach=True)
        return extractor(weights) 
    
    def check_watermark(self, extractor):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        watermark = self.get_watermark(extractor)
        return get_text_from_watermark(watermark)
    
    def get_ber(self, layer_name, extractor, message):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        watermark = self.get_watermark(layer_name, extractor)
        return get_ber(watermark, message)