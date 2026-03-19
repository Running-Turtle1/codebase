import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os


z_dim = 100           # 潜变量维度
image_dim = 28 * 28   # 假设生成 28 * 28 的单通道图像

# 生成器 G
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),


            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),


            nn.Linear(1024, image_dim),
            nn.Tanh() # 将输出结果放到 [-1, 1]
        )
    
    def forward(self, z):
        return self.gen(z)

# 判别器 D
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),


            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid() # 输出 0~1, 代表是真图的概率
        )

    def forward(self, image):
        return self.disc(image)

gen = Generator()
disc = Discriminator()

opt_gen = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))

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

results_dir = 'results_gan'
os.makedirs(results_dir, exist_ok=True)


def save_generated_samples(epoch=None, n=64):
    """保存生成样本，[-1,1] 转 [0,1] 后保存"""
    gen.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim).to(device)
        fake = gen(z)
        fake = (fake + 1) / 2  # [-1,1] -> [0,1]
        fake = fake.view(n, 1, 28, 28)
        path = f'{results_dir}/sample_{epoch}.png' if epoch else f'{results_dir}/final.png'
        save_image(fake, path, nrow=8)
    gen.train()
    return path


for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    epoch_loss_disc = 0.0
    epoch_loss_gen = 0.0
    num_batches = 0

    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size_cur = real_images.shape[0] # 最后一个 batch 可能跟 batch_size 不一样
        real_images = real_images.view(-1, 784).to(device)

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

        epoch_loss_disc += loss_disc.item()
        epoch_loss_gen += loss_gen.item()
        num_batches += 1

    avg_loss_d = epoch_loss_disc / num_batches
    avg_loss_g = epoch_loss_gen / num_batches
    print(f'  D_loss: {avg_loss_d:.4f}  G_loss: {avg_loss_g:.4f}')

    if (epoch + 1) % 10 == 0 or epoch == 0:
        path = save_generated_samples(epoch + 1)
        print(f'  -> Saved: {path}')

# 训练结束，保存最终效果
final_path = save_generated_samples(epoch=None)
print(f'\n===== Training done. Final samples: {final_path} =====')
