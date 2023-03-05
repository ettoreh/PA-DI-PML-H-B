import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import content.dataset_loader.loaders.mnist as dmnist

# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is : ', device)
bs = 1
#train_loader, _ = dmnist.trainset_loader(bs, 0.99, 0, False)
#test_loader = dmnist.testset_loader(bs, 0, False)
# data is model's weights.

class Generator(nn.Module):
    # generator is combined with 4 full-connected layers.
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # define forward
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    # discriminator is combined with 4 full-connected layers.
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # define forward
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))




model_dim =

G = Generator(g_input_dim=z_dim, g_output_dim=model_dim).to(device)
D = Discriminator(model_dim).to(device)
# binary cross entropy
criterion = nn.BCELoss()

# optimiser
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


def D_train(x):
    # ================================================================== #
    #                  Train discriminative model Fdet(θ)                #
    # ================================================================== #
    D.zero_grad()

    # non-watermarked model ，labeled 1
    x_real, y_real = x.view(-1, model_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    # calculate real_loss
    # use BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))，to calculate loss of non-watermarked model
    # the second term id always 0，because real_labels == 1
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # watermarked model ，labeled 0
    z = Variable(torch.randn(bs, z_dim).to(device))
    # generate model embed watermark in the model (which will be Ftgt)
    x_watered, y_watered = G(z), Variable(torch.zeros(bs, 1).to(device))
    # use BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))，to calculate loss of watermarked model
    # the first term is always 0，because y_watered == 0
    D_output = D(x_watered)
    D_fake_loss = criterion(D_output, y_watered)
    D_fake_score = D_output

    # backward and optimise
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x):
    # ================================================================== #
    #                       train Ftgt(w)                     #
    # ================================================================== #
    G.zero_grad()

    # generate model embed watermark in the model（labeled 1）
    y = Variable(torch.ones(bs, 1).to(device))
    G_output = G(w,k)
    # the discriminator distinguish whether has a watermark
    D_output = D(G_output)
    G_loss = uchida_loss-criterion(D_output, y)

    # backward and optimise
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


n_epoch = 10
loss_file = open("loss.txt", 'w')
for epoch in range(1, n_epoch + 1):

    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        # Discard data that is less than the entire batch_size
        if len(x) != bs:
            continue
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))
    loss_file.write(
        '[{}/{}]: loss_d: {:.3f}, loss_g: {:.3f}\n'.format((epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)),
                                                           torch.mean(torch.FloatTensor(G_losses))))
    print("epoch: ", epoch, ' with d_loss ', D_losses, ' with d_loss ', G_losses)
loss_file.close()
