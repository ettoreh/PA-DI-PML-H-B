import torch
import torch.nn as nn

from datetime import datetime

from WideResNet import WideResNet
from load_dataset import trainloader



# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch_index+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) in dropout_index:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
        print("Updated learning rate")



if __name__ == '__main__':
    
    n = 1 
    k = 4
    learning_rate = 0.1
    weight_decay = 5.0*10**(-4)
    momentum = 0.9
    min_batch_size = 64
    dropout_rate = 0.2
    dropout_index = [60, 120, 160]
    num_epochs = 200
    depth = 6+4*n
    
   # Device configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model = WideResNet(depth=10, widen_factor=4, dropout_rate=0.2, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    # Train the model
    total_step = len(trainloader)
    curr_lr = learning_rate
    epoch_number = 0
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(epoch_number)
        
        # Track best performance, and save the model's state
        if (epoch+1)%50 == 0:
            model_path = '/models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
        
    model_path = '/models/model_{}'.format(timestamp)
    torch.save(model.state_dict(), model_path)
        