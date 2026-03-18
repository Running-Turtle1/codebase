import torch
import torch.nn as nn
import torch.nn.functional as F 

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        # Encoder 结构: 输入图片 -> 隐藏层 -> 均值和方差
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)  # 对应均值 mu
        self.fc_logvar = nn.Linear(h_dim, z_dim)  # 对应 log(sigma^2)

        # Decoder 结构: 隐变量 z -> 隐藏层 -> 重构图片
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        # z = mu + eps * sigma
        std = torch.exp(0.5 * logvar)  # sigma = exp(log(sigma^2) / 2)
        eps = torch.randn_like(std)    # 从标准高斯采样噪声
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # 输出重构图片 (0-1) 之间的像素值

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # 1. 重构损失: Binary Cross Entropy
    # x.view(-1, 784) 将 x 展平为 (batch_size, 784)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. KL Divergence: 约束隐空间分布
    # 对应公式 : 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
