import torch 

from .mlpNet.Net import Net
from .wideResNet.WideResNet import WideResNet

    
    
class Network():
    """
    class that allows to load dataset just using the name of one
    """
    def __init__(self, model_name, params, device='cpu') -> None:
        self.model_name = model_name
        self.params = params
        self.device = device
        self.model_list = {
            'mlp': Net,
            'wideResNet': WideResNet
        }
        self.model = self.model_list[self.model_name](*params).to(device)
        pass
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        
    def train(self, num_epoch):
        print('model trained')
    
    def train_wm(self, num_epoch, la):
        print('model')

    


if __name__ == "__main__":
    import sys 
    
    model_name = sys.argv[1]
    params = sys.argv[2]
    params = params.replace('[', '').replace(']', '').split(',')
    params = [int(param) for param in params]
    print(model_name, params)
    
    net = Network(model_name=model_name, params=params, device='mps')
    net.load_model('models/cifar_net_20230216_003445')
    
    