from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from VAE import *

# 1. 超参
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
z_dim = 20
epochs = 15
lr = 1e-3

# 2. dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 3. train
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4. train
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.view(-1, 784))
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


# 5. 测试
def test_generation(epoch):
    model.eval()
    with torch.no_grad():
        # 随机从标准正态分布采样 64 个 z
        z = torch.randn(64, z_dim).to(device)
        sample = model.decode(z).cpu()
        if not os.path.exists('results'):
            os.makedirs('results')
        save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png')

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch)
        test_generation(epoch)