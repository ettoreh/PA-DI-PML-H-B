import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

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
        layers_to_watermark, secret_key, l1, l2,
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
            self._add_watermark(secret_key, layers_to_watermark, l1, l2)
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
        self.ext_optimizer = []
        self.det_optimizer = []
        print('watermark initiated')
        
    # TO DO 
    def _add_watermark(self, secret_key, layers_to_watermark, l1, l2):
        self.to_watermark = True
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
        
        self.layers_to_watermark.append(layers_to_watermark)
        discriminators = {}
        extractors = {}
        ext_optimizers = {}
        det_optimizers = {}
        for name in self.layers_to_watermark[-1]:
            
            discriminators[name] = Discriminator(
                input_size=self.layer_sizes[name],
                hidden_size=2*self.layer_sizes[name]
            ).to(self.device)
            
            extractors[name] = Extractor(
                proportion=0.8,
                input_size=self.layer_sizes[name],
                hidden_size=2*self.layer_sizes[name],
                output_size=len(secret_key),
                device=self.device
            ).to(self.device)
        
            ext_optimizers[name] = torch.optim.Adam(
                extractors[name].parameters(), 
                lr=self.optimizer_params[0]
            )
            
            det_optimizers[name] = torch.optim.Adam(
                discriminators[name].parameters(), 
                lr=self.optimizer_params[0]
            )
        
        self.discriminators.append(discriminators)
        self.extractors.append(extractors)
        self.ext_optimizer.append(ext_optimizers)
        self.det_optimizer.append(det_optimizers)
            
        print('watermark parameters added')
        
    def _train_extractor(self, extractor, optimizer, weights, key, retain_graph=False):
        out = extractor(weights)
        loss = F.binary_cross_entropy_with_logits(out, key)
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        
    # TO DO
    def _train_one_batch(self, batch, watermark_init, weights_init):
        
        torch.autograd.set_detect_anomaly(True)
        
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model(images)
        loss = self.criterion_0(out, labels)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
        if self.to_watermark:
            self.watermarked = True
            
            for name in self.layers_to_watermark[-1]:
                # model weights
                weights = self.get_weights(name, detach=True)
                
                # extractor outputs
                watermark = self.extractors[-1][name](weights)

                # ber with extractor 
                ber = self.get_ber(watermark, self.secret_keys[-1])
                print(ber)
                
                for param in self.extractors[-1][name].parameters():
                    param.requires_grad = True
                
                # extractor training    
                out_ext_wm = self.extractors[-1][name](weights)
                loss_ext_wm = F.binary_cross_entropy_with_logits(out_ext_wm, self.secret_keys[-1])
                out_ext = self.extractors[-1][name](weights_init[name])
                loss_ext = F.binary_cross_entropy_with_logits(out_ext, watermark_init[name])
                loss = loss_ext_wm + loss_ext
                self.ext_optimizer[-1][name].zero_grad()
                loss.backward(retain_graph=True)
                self.ext_optimizer[-1][name].step()
                
                for param in self.extractors[-1][name].parameters():
                    param.requires_grad = False
                    
                for param in self.discriminators[-1][name].parameters():
                    param.requires_grad = True
                    
                # discriminator outputs
                out_det_wm = self.discriminators[-1][name](weights)
                out_det_nwm = self.discriminators[-1][name](weights_init[name])
                print(float(out_det_wm), float(out_det_nwm))
                
                # discriminator training
                loss_det_wm = F.binary_cross_entropy_with_logits(out_det_wm, torch.tensor([1], dtype=torch.float32, device=self.device))
                loss_det_nwm = F.binary_cross_entropy_with_logits(out_det_nwm, torch.tensor([0], dtype=torch.float32, device=self.device))
                loss = loss_det_wm + loss_det_nwm
                self.det_optimizer[-1][name].zero_grad()
                loss.backward(retain_graph=True)
                self.det_optimizer[-1][name].step()
                
                for param in self.discriminators[-1][name].parameters():
                    param.requires_grad = False
                    
                # model training 
                weights = self.get_weights(name)
                watermark = self.extractors[-1][name](weights)
                loss_ext = F.binary_cross_entropy_with_logits(watermark, self.secret_keys[-1])
                loss_det = - torch.log(self.discriminators[-1][name](weights))
                loss = torch.add(loss, loss_ext, alpha=self.l1s[-1])
                loss = torch.add(loss, loss_det, alpha=self.l2s[-1])
        
        # Backward and optimize
        for param in self.model.parameters():
            param.requires_grad = True
            
        print(float(loss))
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
        
        
        
        
        
        
        
        
    def get_weights(self, name, detach=False):
        w = self.model.layers[name].weight.to(torch.float32)
        if detach:
            w = w.detach()
        return torch.mean(w, 0).flatten()
    
    def get_watermark(self, layer_name, extractor):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        weights = self.get_weights(layer_name, detach=True)
        return extractor(weights) 
    
    def check_watermark(self, layer_name, extractor):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        watermark = self.get_watermark(layer_name, extractor)
        return get_text_from_watermark(watermark)
    
    def get_ber(self, watermark, message):
        if not self.watermarked:
            print("the model isn't watermarked")
            return None
        # return get_ber(watermark, message)
        return float(torch.mean((
            (watermark>0.5).to(torch.float32) != message).to(torch.float32), 0))
    
    
    
if __name__=="__main__":
    import sys 
    from content.dataset_loader.datasetLoader import DatasetLoader
    
    model_name = 'net'
    params = [10, [6, 16], 5]
    # print(model_name, params)
    
    # secret_key='Copyright from Ettore Hidoux',
    net = Network(
        model_name=model_name, model_params=params, 
        optimizer='adam', optimizer_params=[0.001, 0.9], 
        layers_to_watermark=['conv1', 'conv2'], secret_key='Copyright from Ettore Hidoux', l1=5, l2=10,
        device=torch.device("mps")
    )
    
    dataset = DatasetLoader(dataset_name='cifar10', batch_size=32, num_workers=4, pin_memory=True)
    trainset, validset = dataset.get_train_valid_loader()
    
    print("training started")
    history = []
    
    weights_init = {}
    watermark_init = {}
    for name in ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']:
        weights_init[name] = net.get_weights(name)
        watermark_init[name] = torch.tensor(get_random_watermark(net.secret_keys[-1].shape[0]), dtype=torch.float32, device=net.device)
            
    for epoch in range(1):
        print("Epoch [{}/{}]".format(epoch+1, 1))
            
        net.model.train(True)
        
        start_time = datetime.now()
        
        training = []
        for i, batch in enumerate(trainset):
            print('--- {} ---'.format(i))
            loss = net._train_one_batch(batch=batch, watermark_init=watermark_init, weights_init=weights_init)
            
            training.append(float(loss))
        