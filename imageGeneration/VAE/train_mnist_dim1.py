"""z_dim=1 的 VAE 训练脚本，1D 潜空间可视化，并分别保存 encoder/decoder 权重"""
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt
from VAE import *

# 1. 超参
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
z_dim = 1        # 1 维潜空间
epochs = 30
lr = 1e-3
weights_dir = 'weights_dim1'
results_dir = 'results_dim1'

# 2. dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 3. model
model = VAE(z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


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


def test_generation(epoch):
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, z_dim).to(device)
        sample = model.decode(z).cpu()
        os.makedirs(results_dir, exist_ok=True)
        save_image(sample.view(64, 1, 28, 28), f'{results_dir}/sample_{epoch}.png')


def save_encoder_decoder_weights():
    os.makedirs(weights_dir, exist_ok=True)
    state = model.state_dict()
    encoder_state = {k: v for k, v in state.items() if k in ('fc1.weight', 'fc1.bias', 'fc_mu.weight', 'fc_mu.bias', 'fc_logvar.weight', 'fc_logvar.bias')}
    decoder_state = {k: v for k, v in state.items() if k in ('fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias')}
    torch.save(encoder_state, os.path.join(weights_dir, 'encoder_z1.pt'))
    torch.save(decoder_state, os.path.join(weights_dir, 'decoder_z1.pt'))
    torch.save(state, os.path.join(weights_dir, 'vae_full_z1.pt'))
    print(f'  -> Saved: {weights_dir}/encoder_z1.pt, decoder_z1.pt, vae_full_z1.pt')


def plot_latent_space(epoch, n=30):
    """1D 潜空间 [-3, 3] 线性采样，生成一行数字"""
    model.eval()
    grid_z = np.linspace(-3, 3, n)
    figure = np.zeros((28, 28 * n))

    with torch.no_grad():
        for j, zi in enumerate(grid_z):
            z_sample = torch.FloatTensor([[zi]]).to(device)
            x_decoded = model.decode(z_sample).cpu().numpy()
            digit = x_decoded[0].reshape(28, 28)
            figure[:, j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(12, 2))
    plt.imshow(figure, cmap='Greys_r', aspect='auto')
    plt.axis('off')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/latent_line_{epoch}.png', bbox_inches='tight', dpi=100)
    plt.close()
    print(f'  -> Saved latent line: {results_dir}/latent_line_{epoch}.png')


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch)
        test_generation(epoch)
        if epoch % 5 == 0 or epoch == 1:
            plot_latent_space(epoch)
        scheduler.step()

    save_encoder_decoder_weights()
