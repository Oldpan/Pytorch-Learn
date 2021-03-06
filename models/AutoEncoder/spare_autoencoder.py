import torch
from torch import nn, optim

from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import dataloader

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

device = torch.device('cuda:0')

train_data = MNIST(root='mnist', train=True, transform=transforms.ToTensor())

configure = {
    'epoch': 20,
    'batch_size': 64,
    'lr': 0.005,
}

print('MNIST data size:', train_data.train_data.size())      # (60000, 28, 28)
print('MNIST data label:', train_data.train_labels.size())   # (60000)

plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()

data_loader = dataloader.DataLoader(dataset=train_data, batch_size=configure['batch_size'], shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 28*28),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(28*28, 28*28),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(28*28, 28*28),
        )
        self.decoder = nn.Sequential(
            nn.Linear(28*28, 28*28),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(28*28, 28*28),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(28*28, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        return hidden, output


autoencoder = AutoEncoder().to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=configure['lr'], weight_decay=1e-4)

loss_f = nn.MSELoss()

f, a = plt.subplots(2, 5, figsize=(5, 2))
# plt.ion()

# original data (first row) for viewing
view_data = train_data.train_data[:5].view(-1, 28*28).float()/255.

for i in range(5):
    a[0][i].set_title('{}'.format(train_data.train_labels[i]))
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())


for epoch in range(configure['epoch']):
    for step, (x, y) in enumerate(data_loader):

        x = x.to(device)
        y = y.to(device)
        b_x = x.view(-1, 28*28)   # batch x, shape (batch, 28*28)
        b_y = x.view(-1, 28*28)   # batch y, shape (batch, 28*28)
        b_label = y               # batch label

        hidden, output = autoencoder(b_x)

        loss = loss_f(output, b_y)
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| Loss: %.4f' % loss.item())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data.to(device))
            for i in range(5):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

            plt.imshow(np.reshape(decoded_data.cpu().data.numpy()[0], (28, 28)), cmap='gray')
            plt.pause(0.05)

            # plt.draw()
            # plt.pause(0.05)

plt.ioff()
plt.show()


