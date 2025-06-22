# 导入所需的Python包
import random  # 引入随机数生成模块
import numpy as np  # 引入NumPy数值计算库
import torch  # 引入PyTorch深度学习框架
from torch import nn  # 引入神经网络模块
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset  # 引入数据加载工具
import torchvision.transforms as transforms  # 引入图像变换工具
import torch.nn.functional as F  # 引入函数式接口
from torch.autograd import Variable  # 引入自动微分变量
import torchvision.models as models  # 引入预训练模型
from torch.optim import Adam, AdamW  # 引入优化器
from qqdm import qqdm, format_str  # 引入进度条显示工具
import pandas as pd  # 引入数据处理库

# 加载数据集
train = np.load('data/trainingset.npy', allow_pickle=True)  # 加载训练集
test = np.load('data/testingset.npy', allow_pickle=True)  # 加载测试集

print(train.shape)  # 打印训练集形状
print(test.shape)  # 打印测试集形状

# 设置随机种子函数，确保实验可重复性
def same_seeds(seed):
    random.seed(seed)  # 设置Python随机数种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    if torch.cuda.is_available():  # 如果有GPU可用
        torch.cuda.manual_seed(seed)  # 设置当前GPU随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机数种子
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的自动优化
    torch.backends.cudnn.deterministic = True  # 确保cudnn的结果是确定的

same_seeds(20030908)  # 使用具体的种子值调用上面的函数

# 自动编码器模型定义
# 全连接自动编码器模型
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # 定义编码器部分
            nn.Linear(64 * 64 * 3, 128),  # 输入层到128维隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 64),  # 128维到64维
            nn.ReLU(), 
            nn.Linear(64, 12),  # 64维到12维
            nn.ReLU(), 
            nn.Linear(12, 3)  # 12维到3维潜在空间
        )
        
        self.decoder = nn.Sequential(  # 定义解码器部分
            nn.Linear(3, 12),  # 3维潜在空间到12维
            nn.ReLU(), 
            nn.Linear(12, 64),  # 12维到64维
            nn.ReLU(),
            nn.Linear(64, 128),  # 64维到128维
            nn.ReLU(), 
            nn.Linear(128, 64 * 64 * 3),  # 128维到原始图像维度
            nn.Tanh()  # 输出范围限制在[-1,1]
        )

    def forward(self, x):
        x = self.encoder(x)  # 通过编码器
        x = self.decoder(x)  # 通过解码器
        return x


# 卷积自动编码器模型
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # 定义编码器部分
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # 第一个卷积层，3通道→12通道
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # 第二个卷积层，12通道→24通道
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # 第三个卷积层，24通道→48通道
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(  # 定义解码器部分
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # 第一个反卷积层，48通道→24通道
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # 第二个反卷积层，24通道→12通道
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # 第三个反卷积层，12通道→3通道
            nn.Tanh(),  # 输出范围限制在[-1,1]
        )

    def forward(self, x):
        x = self.encoder(x)  # 通过编码器
        x = self.decoder(x)  # 通过解码器
        return x


# 变分自动编码器(VAE)模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(  # 定义共享编码器部分
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # 第一个卷积层           
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # 第二个卷积层   
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(  # 均值编码器分支
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # 输出均值μ
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(  # 对数方差编码器分支
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # 输出对数方差logσ²
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(  # 定义解码器部分
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # 第一个反卷积层
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # 第二个反卷积层
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # 第三个反卷积层
            nn.Tanh(),  # 输出范围限制在[-1,1]
        )

    def encode(self, x):
        h1 = self.encoder(x)  # 通过共享编码器
        return self.enc_out_1(h1), self.enc_out_2(h1)  # 返回均值和对数方差

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 在GPU上生成正态分布随机数
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # 在CPU上生成正态分布随机数
        eps = Variable(eps)  # 转换为Variable
        return eps.mul(std).add_(mu)  # 重参数化技巧：z = μ + σ * ε

    def decode(self, z):
        return self.decoder(z)  # 通过解码器

    def forward(self, x):
        mu, logvar = self.encode(x)  # 获取均值和对数方差
        z = self.reparametrize(mu, logvar)  # 重参数化采样
        return self.decode(z), mu, logvar  # 返回重建图像、均值和对数方差


# VAE的损失函数
def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: 重建的图像
    x: 原始图像
    mu: 潜在空间均值
    logvar: 潜在空间对数方差
    """
    mse = criterion(recon_x, x)  # 计算重建误差(MSE)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)  # KL散度计算元素
    KLD = torch.sum(KLD_element).mul_(-0.5)  # 计算总KL散度
    return mse + KLD  # 返回总损失：重建误差+KL散度

# 自定义数据集类
class CustomTensorDataset(TensorDataset):
    """支持转换功能的TensorDataset
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:  # 如果最后一维是3（即RGB通道）
            self.tensors = tensors.permute(0, 3, 1, 2)  # 调整维度顺序为PyTorch标准格式(B,C,H,W)
        
        self.transform = transforms.Compose([  # 定义转换操作
          transforms.Lambda(lambda x: x.to(torch.float32)),  # 转换为float32类型
          transforms.Lambda(lambda x: 2. * x/255. - 1.),  # 归一化到[-1,1]范围
        ])
        
    def __getitem__(self, index):
        x = self.tensors[index]  # 获取指定索引的样本
        
        if self.transform:
            # 将图像映射到[-1.0, 1.0]范围
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)  # 返回数据集长度
    
# 训练配置
# 训练超参数
num_epochs = 50  # 训练轮数
batch_size = 2000  # 批次大小
learning_rate = 1e-3  # 学习率

# 构建训练数据加载器
x = torch.from_numpy(train)  # 将NumPy数组转换为PyTorch张量
train_dataset = CustomTensorDataset(x)  # 创建训练数据集

train_sampler = RandomSampler(train_dataset)  # 创建随机采样器
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)  # 创建训练数据加载器

# 模型选择
model_type = 'vae'   # 从{'cnn', 'fcn', 'vae'}中选择一个模型类型
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE()}  # 模型类型映射字典
model = model_classes[model_type].cuda()  # 实例化选择的模型并移至GPU

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练循环

best_loss = np.inf  # 初始化最佳损失为无穷大
model.train()  # 设置模型为训练模式

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))  # 创建进度条
for epoch in qqdm_train:
    tot_loss = list()  # 存储每个批次的损失
    for data in train_dataloader:

        # ===================加载数据=====================
        img = data.float().cuda()  # 将数据转换为float并移至GPU
        if model_type in ['fcn']:  # 如果是全连接网络模型
            img = img.view(img.shape[0], -1)  # 将图像展平

        # ===================前向传播=====================
        output = model(img)  # 模型前向传播
        if model_type in ['vae']:  # 如果是VAE模型
            loss = loss_vae(output[0], img, output[1], output[2], criterion)  # 使用VAE特定的损失函数
        else:  # 其他模型
            loss = criterion(output, img)  # 使用MSE损失

        tot_loss.append(loss.item())  # 记录损失值
        # ===================反向传播====================
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
    # ===================保存最佳模型====================
    mean_loss = np.mean(tot_loss)  # 计算平均损失
    if mean_loss < best_loss:  # 如果当前损失小于最佳损失
        best_loss = mean_loss  # 更新最佳损失
        torch.save(model, 'best_model_{}.pt'.format(model_type))  # 保存最佳模型
    # ===================记录信息========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',  # 当前轮次/总轮次
        'loss': f'{mean_loss:.4f}',  # 平均损失
    })
    # ===================保存最终模型========================
    torch.save(model, 'last_model_{}.pt'.format(model_type))  # 保存每轮的最终模型


# 推理阶段
# 初始化
eval_batch_size = 200  # 评估批次大小

# 构建测试数据加载器
data = torch.tensor(test, dtype=torch.float32)  # 将测试数据转换为PyTorch张量
test_dataset = CustomTensorDataset(data)  # 创建测试数据集
test_sampler = SequentialSampler(test_dataset)  # 创建顺序采样器
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=0)
eval_loss = nn.MSELoss(reduction='none')  # 用于评估的MSE损失，不进行求和操作

# 加载训练好的模型
checkpoint_path = f'last_model_{model_type}.pt'  # 模型路径
model = torch.load(checkpoint_path)  # 加载模型
model.eval()  # 设置为评估模式

# 预测文件
out_file = 'prediction.csv'  # 输出文件名

anomality = list()  # 存储异常分数
with torch.no_grad():  # 不计算梯度以提高效率
  for i, data in enumerate(test_dataloader):
    img = data.float().cuda()  # 将数据转换为float并移至GPU
    if model_type in ['fcn']:  # 如果是全连接网络
      img = img.view(img.shape[0], -1)  # 展平图像
    output = model(img)  # 模型前向传播
    if model_type in ['vae']:  # 如果是VAE
      output = output[0]  # 只取重建输出
    if model_type in ['fcn']:  # 如果是全连接网络
        loss = eval_loss(output, img).sum(-1)  # 沿最后一维求和得到每个样本的重建误差
    else:  # 卷积网络或VAE
        loss = eval_loss(output, img).sum([1, 2, 3])  # 沿通道、高度、宽度维度求和
    anomality.append(loss)  # 添加到异常分数列表
anomality = torch.cat(anomality, axis=0)  # 合并所有批次的异常分数
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()  # 计算均方根误差并转换格式

df = pd.DataFrame(anomality, columns=['score'])  # 创建包含异常分数的DataFrame
df.to_csv(out_file, index_label = 'ID')  # 保存为CSV文件，以ID为索引
