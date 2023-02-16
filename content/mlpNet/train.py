import torch
import torch.nn as nn

from datetime import datetime

from Net import Net
from load_dataset import trainloader



def train_one_epoch(epoch_index):

    for i, (images, labels) in enumerate(trainloader):
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



if __name__ == '__main__':
    
    learning_rate = 0.001
    momentum = 0.9
    min_batch_size = 4
    num_epochs = 10 #3
    
    # Device configuration
    # device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model = Net(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

    # Train the model
    total_step = len(trainloader)
    curr_lr = learning_rate
    epoch_number = 0
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        start = datetime.now()
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(epoch_number)
        
        epoch_number += 1
        end = datetime.now()
        print("EPOCH time = {}".format(end-start))
    
    path = '/Users/ettorehidoux/Desktop/codes projects/PA-DI-PML-H-B/models'
    name = '/cifar_net_{}'.format(timestamp)
    torch.save(model.state_dict(), path+name)
        