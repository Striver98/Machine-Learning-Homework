# ========================================
# 导入必要的库
# ========================================
import numpy as np  # 数值计算库
import random       # 随机数生成
import torch        # PyTorch深度学习框架

# PyTorch数据处理相关模块
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torchvision.transforms as transforms  # 图像变换

# PyTorch神经网络相关模块
from torch import nn                    # 神经网络层
import torch.nn.functional as F         # 神经网络函数
from torch.autograd import Variable     # 自动梯度变量
import torchvision.models as models     # 预训练模型

# 优化器
from torch.optim import Adam, AdamW

# 辅助库
from qqdm import qqdm, format_str       # 进度条显示
import pandas as pd                     # 数据处理

import pdb  # 调试工具，使用pdb.set_trace()设置断点

# ========================================
# 数据加载
# ========================================
# 加载训练和测试数据集（.npy格式的numpy数组文件）
train = np.load('data/trainingset.npy', allow_pickle=True)  # 加载训练集
test = np.load('data/testingset.npy', allow_pickle=True)    # 加载测试集

print(train.shape)  # 打印训练集形状，例如：(10000, 64, 64, 3)
print(test.shape)   # 打印测试集形状，例如：(2000, 64, 64, 3)

# ========================================
# 随机种子设置函数
# ========================================
def same_seeds(seed):
    """
    设置所有随机数生成器的种子，确保实验结果可重现
    Args:
        seed: 随机种子值
    """
    # Python内置随机模块
    random.seed(seed)
    # Numpy随机模块
    np.random.seed(seed)
    # PyTorch随机模块
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # 设置当前GPU的随机种子
        torch.cuda.manual_seed_all(seed)     # 设置所有GPU的随机种子
    torch.backends.cudnn.benchmark = False   # 禁用cudnn的自动优化，确保结果一致
    torch.backends.cudnn.deterministic = True # 使cudnn使用确定性算法

same_seeds(20030908)  # 设置随机种子

# ========================================
# 模型定义
# ========================================

# 1. 全连接自编码器（FCN Autoencoder）
class fcn_autoencoder(nn.Module):
    """
    全连接自编码器
    将64x64x3的图像压缩到3维潜在空间，然后重建
    """
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        
        # 编码器：将12288维（64*64*3）压缩到3维
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 2048),
            nn.ReLU(), 
            nn.Linear(2048, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, 512), 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64 * 64 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)     # 编码
        x = self.decoder(x)     # 解码
        return x

# 2. 卷积自编码器（Convolutional Autoencoder）
class conv_autoencoder(nn.Module):
    """
    卷积自编码器
    使用卷积和转置卷积处理图像数据
    """
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        
        # 编码器：通过卷积层逐步降采样
        self.encoder = nn.Sequential(
            # 第一层：3通道 -> 128通道，尺寸减半（64->32）
            nn.Conv2d(3, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),            # 批量归一化
            nn.ReLU(),
            # 第二层：128通道 -> 256通道，尺寸减半（32->16）
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 第三层：256通道 -> 512通道，尺寸减半（16->8）
            nn.Conv2d(256, 512, 4, stride=2, padding=1),   
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # 解码器：通过转置卷积逐步上采样
        self.decoder = nn.Sequential(
            # 第一层：512通道 -> 256通道，尺寸翻倍（8->16）
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 第二层：256通道 -> 128通道，尺寸翻倍（16->32）
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第三层：128通道 -> 3通道，尺寸翻倍（32->64）
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1), 
            nn.Tanh(),                      # 输出激活，范围[-1, 1]
        )

    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)     # 编码
        x = self.decoder(x)     # 解码
        return x

# 3. 变分自编码器（Variational Autoencoder, VAE）
class VAE(nn.Module):
    """
    变分自编码器
    学习数据的概率分布，而不是确定性编码
    """
    def __init__(self):
        super(VAE, self).__init__()
        
        # 共享编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),    # 64->32，3->12通道
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),   # 32->16，12->24通道
            nn.ReLU(),
        )
        
        # 均值分支（mu）
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),   # 16->8，24->48通道
            nn.ReLU(),
        )
        
        # 方差分支（logvar）
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),   # 16->8，24->48通道
            nn.ReLU(),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # 32->64
            nn.Tanh(),
        )

    def encode(self, x):
        """编码过程，返回均值和对数方差"""
        h1 = self.encoder(x)        # 共享特征提取
        return self.enc_out_1(h1), self.enc_out_2(h1)  # 返回mu和logvar

    def reparametrize(self, mu, logvar):
        """
        重参数化技巧：从高斯分布中采样
        z = mu + sigma * epsilon，其中epsilon ~ N(0,1)
        """
        std = logvar.mul(0.5).exp_()    # 计算标准差：std = exp(logvar/2)
        
        # 生成随机噪声
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)    # z = mu + std * eps

    def decode(self, z):
        """解码过程"""
        return self.decoder(z)

    def forward(self, x):
        """VAE前向传播"""
        mu, logvar = self.encode(x)                     # 编码得到mu和logvar
        z = self.reparametrize(mu, logvar)              # 重参数化采样
        return self.decode(z), mu, logvar               # 返回重建结果、均值、对数方差

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    VAE损失函数：重建损失 + KL散度
    Args:
        recon_x: 重建图像
        x: 原始图像
        mu: 潜在变量均值
        logvar: 潜在变量对数方差
        criterion: 重建损失函数（如MSE）
    """
    mse = criterion(recon_x, x)  # 重建损失（MSE）
    
    # KL散度：衡量学习到的分布与标准正态分布的差异
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    
    return mse + KLD  # 总损失 = 重建损失 + KL散度

# 4. ResNet自编码器
class Resnet(nn.Module):
    """
    基于ResNet的自编码器
    使用预训练的ResNet作为编码器骨干
    """
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(Resnet, self).__init__()

        # 保存超参数
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # 卷积层参数（用于解码器）
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 卷积核大小
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 步长
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 填充

        # 编码器组件：使用ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=False)      # 不使用预训练权重
        modules = list(resnet.children())[:-1]          # 移除最后的全连接层
        self.resnet = nn.Sequential(*modules)           # ResNet特征提取器
        
        # 全连接层将ResNet输出映射到潜在空间
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # 潜在表示

        # 解码器：从潜在空间重建图像
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)  # 重建为4x4特征图
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # 转置卷积层逐步上采样
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                             kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, 
                             kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, 
                             kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # 输出范围[0,1]
        )

    def encode(self, x):
        """编码过程"""
        x = self.resnet(x)                  # ResNet特征提取
        x = x.view(x.size(0), -1)           # 展平

        # 全连接层（考虑批量大小为1的情况）
        if x.shape[0] > 1:
            x = self.bn1(self.fc1(x))       # 使用批量归一化
        else:
            x = self.fc1(x)                 # 不使用批量归一化
        x = self.relu(x)
        
        if x.shape[0] > 1:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_mu(x)                  # 潜在表示
        return x

    def decode(self, z):
        """解码过程"""
        # 全连接层
        if z.shape[0] > 1:
            x = self.relu(self.fc_bn4(self.fc4(z)))
            x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        else:
            x = self.relu(self.fc4(z))
            x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)
        
        # 转置卷积层
        x = self.convTrans6(x)              # 4x4 -> 8x8
        x = self.convTrans7(x)              # 8x8 -> 16x16
        x = self.convTrans8(x)              # 16x16 -> 32x32
        
        # 双线性插值上采样到目标尺寸
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        """前向传播"""
        z = self.encode(x)                  # 编码
        x_reconst = self.decode(z)          # 解码
        return x_reconst

# ========================================
# 自定义数据集类
# ========================================
class CustomTensorDataset(TensorDataset):
    """
    支持变换的张量数据集
    """
    def __init__(self, tensors):
        self.tensors = tensors
        
        # 如果输入格式是(N, H, W, C)，转换为(N, C, H, W)
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        # 定义数据变换：转换为float32并归一化到[-1, 1]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),   # 转换数据类型
            transforms.Lambda(lambda x: 2. * x/255. - 1.),     # 归一化到[-1, 1]
        ])
        
    def __getitem__(self, index):
        """获取单个样本"""
        x = self.tensors[index]
        
        if self.transform:
            x = self.transform(x)   # 应用变换
        
        return x

    def __len__(self):
        """返回数据集大小"""
        return len(self.tensors)

# ========================================
# 训练设置
# ========================================
# 训练超参数
num_epochs = 100        # 训练轮数
batch_size = 128        # 批次大小
learning_rate = 1e-3    # 学习率

# 构建训练数据加载器
x = torch.from_numpy(train)                         # 转换为PyTorch张量
train_dataset = CustomTensorDataset(x)              # 创建数据集
train_sampler = RandomSampler(train_dataset)        # 随机采样器
train_dataloader = DataLoader(train_dataset, 
                             sampler=train_sampler, 
                             batch_size=batch_size)  # 数据加载器

# 模型选择和初始化
model_type = 'fcn'  # 从{'cnn', 'fcn', 'vae', 'resnet'}中选择模型类型
model_classes = {
    'resnet': Resnet(), 
    'fcn': fcn_autoencoder(), 
    'cnn': conv_autoencoder(), 
    'vae': VAE()
}
model = model_classes[model_type].cuda()    # 将模型移动到GPU

# 损失函数和优化器
criterion = nn.MSELoss()                    # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# ========================================
# 训练循环
# ========================================
best_loss = np.inf      # 初始化最佳损失为无穷大
model.train()           # 设置模型为训练模式

# 创建进度条
qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))

for epoch in qqdm_train:
    tot_loss = list()   # 存储每个批次的损失
    
    for data in train_dataloader:
        # ===================加载数据=====================
        img = data.float().cuda()  # 将数据转换为float并移至GPU
        
        # 如果是全连接网络，需要将图像展平
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)  # 展平：(B, C, H, W) -> (B, C*H*W)

        # ===================前向传播=====================
        output = model(img)  # 模型前向传播
        
        # 计算损失
        if model_type in ['vae']:  # VAE使用特殊损失函数
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:  # 其他模型使用MSE损失
            loss = criterion(output, img)

        tot_loss.append(loss.item())  # 记录损失值
        
        # ===================反向传播====================
        optimizer.zero_grad()   # 清除之前的梯度
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 更新模型参数
    
    # ===================保存最佳模型====================
    mean_loss = np.mean(tot_loss)  # 计算平均损失
    if mean_loss < best_loss:      # 如果当前损失小于最佳损失
        best_loss = mean_loss      # 更新最佳损失
        torch.save(model, 'best_model_{}.pt'.format(model_type))  # 保存最佳模型
    
    # ===================记录信息========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',  # 当前轮次/总轮次
        'loss': f'{mean_loss:.4f}',                     # 平均损失
    })
    
    # 保存每轮的最终模型
    torch.save(model, 'last_model_{}.pt'.format(model_type))

# ========================================
# 推理阶段（异常检测）
# ========================================
# 初始化
eval_batch_size = 200  # 评估批次大小

# 构建测试数据加载器
data = torch.tensor(test, dtype=torch.float32)         # 转换测试数据
test_dataset = CustomTensorDataset(data)               # 创建测试数据集
test_sampler = SequentialSampler(test_dataset)         # 顺序采样器
test_dataloader = DataLoader(test_dataset, 
                           sampler=test_sampler, 
                           batch_size=eval_batch_size, 
                           num_workers=0)

# 用于评估的损失函数（不进行reduction）
eval_loss = nn.MSELoss(reduction='none')

# 加载训练好的模型
checkpoint_path = f'last_model_{model_type}.pt'
model = torch.load(checkpoint_path)
model.eval()  # 设置为评估模式

# 预测输出文件
out_file = 'prediction.csv'

# 计算异常分数
anomality = list()  # 存储异常分数
with torch.no_grad():  # 不计算梯度以提高效率和节省内存
    for i, data in enumerate(test_dataloader):
        img = data.float().cuda()  # 移动到GPU
        
        # 处理全连接网络的输入格式
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)  # 展平图像
        
        # 模型前向传播
        output = model(img)
        
        # 处理VAE的输出格式
        if model_type in ['vae']:
            output = output[0]  # VAE返回(重建, mu, logvar)，只取重建结果
        
        # 计算重建误差
        if model_type in ['fcn']:  # 全连接网络
            # 沿最后一维求和得到每个样本的重建误差
            loss = eval_loss(output, img).sum(-1)
        else:  # 卷积网络或VAE
            # 沿通道、高度、宽度维度求和
            loss = eval_loss(output, img).sum([1, 2, 3])
        
        anomality.append(loss)  # 添加到异常分数列表

# 后处理异常分数
anomality = torch.cat(anomality, axis=0)  # 合并所有批次的异常分数
# 计算均方根误差作为最终异常分数
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

# 保存结果
df = pd.DataFrame(anomality, columns=['score'])  # 创建DataFrame
df.to_csv(out_file, index_label='ID')           # 保存为CSV文件，以ID为索引标签

print(f"异常检测完成！结果已保存到 {out_file}")
