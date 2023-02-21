import torch 
import torch.nn as nn 
import numpy as np

from watermark.custom_loss import watermarkCrossEntropyLoss



def train_one_epoch(
    epoch_index, num_epochs, total_step, model, dataset, watermark, la, criterion_0, criterion_r, optimizer, device):
    
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        weights = model.conv1.weight.detach().cpu().numpy()
        
        loss_0 = criterion_0(outputs, labels).to(torch.float32)
        loss_r = criterion_r(weights, watermark).to(torch.float32)
        
        loss = torch.add(loss_0, loss_r, alpha=la, out=None)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch_index+1, num_epochs, i+1, total_step, loss.item()))

def train(num_epochs, model, model_path, dataset, watermark, type, size, matrix_path, la, lr, mom, device):

    criterion_0 = nn.CrossEntropyLoss()
    criterion_r = watermarkCrossEntropyLoss(type=type, size=size)
    print(criterion_r.X)
    criterion_r.save(matrix_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, nesterov=True)

    # Train the model
    total_step = len(dataset)
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        start = datetime.now()
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(
            epoch, num_epochs, total_step, model, dataset, watermark, la, criterion_0, criterion_r, optimizer, device)
        
        end = datetime.now()
        print("EPOCH time = {}".format(end-start))
        
    torch.save(model.state_dict(), model_path)
    
    

if __name__ == '__main__':
    import sys
    
    from datetime import datetime
    from dataset_loader.datasetLoader import DatasetLoader
    from networks.networks import Network
    from watermark.secret_matrix import get_watermark_from_text
    


    model = sys.argv[1]
    data = sys.argv[2]
    params = sys.argv[3]
    params = params.replace('[', '').replace(']', '').split(',')
    params = [int(param) for param in params]
    epoch = int(sys.argv[4])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/' 
    matrix_path = model_path + 'models/matrix_' + timestamp +'.npy'
    model_path = model_path + 'models/' + model + '_' + data + '_' + timestamp
    print(model)
    print(data)
    print(params)
    print(model_path)
    print(matrix_path)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('on machine ', device)
    
    # model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models/cifar_net_20230216_003445'
    model = Network(model_name=model, params=params, device=device)
    
    dataset = DatasetLoader(dataset_name=data, batch_size=4)
    train_data = dataset.trainset_loader()
    
    watermark = get_watermark_from_text('Ettore Hidoux')
    M = len(np.mean(model.model.conv1.weight.detach().cpu().numpy(), 0).flatten())
    T = len(watermark)
    watermark = torch.tensor(watermark, dtype=torch.float32)
    
    print(watermark)
    print(M, T)
        
    train(
        num_epochs=epoch, model=model.model, model_path=model_path, 
        dataset=train_data, watermark=watermark, type='direct', size=(T, M), 
        matrix_path=matrix_path, la=10, lr=0.001, mom=0.9, device=device
    )
    