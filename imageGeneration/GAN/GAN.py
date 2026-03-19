import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms


z_dim = 100           # 潜变量维度
image_dim = 28 * 28   # 假设生成 28 * 28 的单通道图像

# 生成器 G
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, image_dim),
            nn.Tanh() # 将输出结果放到 [-1, 1]
        )
    
    def forward(self, z):
        return self.gen(z)

# 判别器 D
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # 输出 0~1, 代表是真图的概率
        )

    def forward(self, image):
        return self.disc(image)

gen = Generator()
disc = Discriminator()

opt_gen = optim.Adam(gen.parameters(), lr=0.0002)
opt_disc = optim.Adam(disc.parameters(), lr=0.0002)

criterion = nn.BCELoss()

num_epochs = 50
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0,1] -> [-1,1]，与 G 的 Tanh 输出一致
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

gen = gen.to(device)
disc = disc.to(device)

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size_cur = real_images.shape[0]
        real_images = real_images.view(-1, 784).to(device)

        # 随机噪声
        z = torch.randn(batch_size_cur, z_dim).to(device)
        fake_images = gen(z)

        # 1. train D
        # 目标: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real_images)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake_images.detach())
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # 2. train G
        # 目标: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake_images)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
