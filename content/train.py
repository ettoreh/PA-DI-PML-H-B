import torch 
import torch.nn as nn 



def train_one_epoch(
    epoch_index, num_epochs, total_step, model, dataset, criterion, optimizer, device):
    
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch_index+1, num_epochs, i+1, total_step, loss.item()))

def train(num_epochs, model, dataset, lr, mom, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, nesterov=True)

    # Train the model
    total_step = len(dataset)
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        start = datetime.now()
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(
            epoch, num_epochs, total_step, model, dataset, criterion, optimizer, device)
        
        end = datetime.now()
        print("EPOCH time = {}".format(end-start))
        
    torch.save(model.state_dict(), model_path)
    
    

if __name__ == '__main__':
    import sys
    
    from datetime import datetime
    from dataset_loader.datasetLoader import DatasetLoader
    from networks.networks import Network
    


    model = sys.argv[1]
    data = sys.argv[2]
    params = sys.argv[3]
    params = params.replace('[', '').replace(']', '').split(',')
    params = [int(param) for param in params]
    epoch = int(sys.argv[4])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/' 
    model_path = model_path + 'models/' + model + '_' + data + '_' + timestamp
    print(model)
    print(data)
    print(params)
    print(model_path)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('on machine ', device)
    
    # model_path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models/cifar_net_20230216_003445'
    model = Network(model_name=model, params=params, device=device)
    
    dataset = DatasetLoader(dataset_name=data, batch_size=4)
    train_data = dataset.trainset_loader()
    
    train(num_epochs=epoch, model=model.model, dataset=train_data, lr=0.001, mom=0.9, device=device)
    