# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn

# 设置计算设备：如果有GPU则使用CUDA，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置批处理大小为8，即每次处理8张图片
batch_size = 8

# CIFAR-10数据集预计算的均值和标准差统计值
cifar_10_mean = (0.491, 0.482, 0.447) # CIFAR-10图像三个通道(RGB)的均值
cifar_10_std = (0.202, 0.199, 0.201) # CIFAR-10图像三个通道(RGB)的标准差

# 将均值和标准差转换为3维张量，形状为(3,1,1)，便于后续广播运算
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

# 计算扰动限制epsilon：8/255除以标准差，这是对抗攻击中允许的最大扰动幅度
epsilon = 8/255/std

# 设置数据根目录，存储原始良性图像
root = './data' # 存储良性图像的目录
# 良性图像：不包含对抗扰动的原始图像
# 对抗图像：包含对抗扰动的图像

# 导入必要的库
import os          # 操作系统接口
import glob        # 文件路径模式匹配
import shutil      # 高级文件操作
import numpy as np # 数值计算库
from PIL import Image  # 图像处理库
from torchvision.transforms import transforms  # 图像变换
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 定义图像预处理变换序列
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为张量，像素值从[0,255]缩放到[0,1]
    transforms.Normalize(cifar_10_mean, cifar_10_std)  # 使用CIFAR-10的均值和标准差进行标准化
])

# 定义对抗数据集类，继承自PyTorch的Dataset类
class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        # 初始化三个列表存储图像路径、标签和名称
        self.images = []  # 存储图像文件的完整路径
        self.labels = []  # 存储对应的类别标签
        self.names = []   # 存储相对路径名称
        '''
        数据目录结构：
        data_dir
        ├── class_dir (类别目录)
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        # 遍历数据目录下的所有类别文件夹
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            # 获取当前类别文件夹下的所有图像文件，并排序
            images = sorted(glob.glob(f'{class_dir}/*'))
            # 将图像路径添加到列表中
            self.images += images
            # 为当前类别的所有图像分配标签i（文件夹索引作为类别标签）
            self.labels += ([i] * len(images))
            # 生成相对路径名称并添加到列表中
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        # 保存变换函数
        self.transform = transform
    
    def __getitem__(self, idx):
        # 根据索引获取数据项：打开图像，应用变换，返回图像和标签
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    
    def __getname__(self):
        # 返回所有图像的相对路径名称
        return self.names
    
    def __len__(self):
        # 返回数据集的大小（图像总数）
        return len(self.images)

# 创建对抗数据集实例
adv_set = AdvDataset(root, transform=transform)
# 获取所有图像的名称
adv_names = adv_set.__getname__()
# 创建数据加载器，不打乱顺序
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

# 打印数据集中图像的总数
print(f'number of images = {adv_set.__len__()}')

# 定义函数：评估模型在良性图像上的性能
def epoch_benign(model, loader, loss_fn):
    model.eval()  # 设置模型为评估模式，关闭dropout等训练时的随机性
    train_acc, train_loss = 0.0, 0.0  # 初始化准确率和损失累计器
    # 遍历数据加载器中的每个批次
    for x, y in loader:
        x, y = x.to(device), y.to(device)  # 将数据移动到指定设备（GPU/CPU）
        yp = model(x)  # 前向传播，获取模型预测
        loss = loss_fn(yp, y)  # 计算损失
        # 累计正确预测的数量：比较预测类别与真实标签
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        # 累计总损失：乘以批次大小得到总损失
        train_loss += loss.item() * x.shape[0]
    # 返回平均准确率和平均损失
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# 定义FGSM（Fast Gradient Sign Method）对抗攻击函数
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone()  # 将原始良性图像x复制作为对抗样本的初始值
    x_adv.requires_grad = True  # 设置需要计算梯度，因为要对输入求梯度
    loss = loss_fn(model(x_adv), y)  # 计算模型在当前输入上的损失
    loss.backward()  # 反向传播计算梯度
    # FGSM攻击：使用梯度上升来最大化损失函数
    grad = x_adv.grad.detach()  # 获取梯度并分离计算图
    x_adv = x_adv + epsilon * grad.sign()  # 沿梯度符号方向添加扰动
    return x_adv

# 设置I-FGSM的步长：可以自行调整
alpha = 0.8/255/std
# 定义I-FGSM（Iterative FGSM）对抗攻击函数
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x  # 初始化对抗样本为原始图像
    # 执行num_iter次迭代攻击
    for i in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # 也可以调用fgsm函数，用alpha作为扰动大小
        x_adv = x_adv.detach().clone()  # 分离梯度并复制
        x_adv.requires_grad = True  # 设置需要计算梯度
        loss = loss_fn(model(x_adv), y)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        # 执行FGSM攻击：使用梯度上升最大化损失
        grad = x_adv.grad.detach()  # 获取梯度
        x_adv = x_adv + alpha * grad.sign()  # 添加小步长扰动
        
        # 将新的对抗样本裁剪回允许的扰动范围[x-epsilon, x+epsilon]
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon)
    return x_adv

# MI-FGSM（Momentum Iterative FGSM）攻击
def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0):
    x_adv = x
    # 初始化动量张量
    momentum = torch.zeros_like(x).detach().to(device)
    # 执行num_iter次迭代
    grad = 0
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        # TODO: 动量计算
        grad =  decay * grad + x_adv.grad.detach() / torch.norm(x_adv.grad.detach())
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon)
    return x_adv

# DIM-MIFGSM (Diverse Input Method MI-FGSM) 攻击
def dim_mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0, prob=0.5):
    """
    DIM-MIFGSM攻击函数，结合了动量和多样化输入变换
    Args:
        model: 目标模型
        x: 输入图像
        y: 真实标签
        loss_fn: 损失函数
        epsilon: 扰动范围
        alpha: 步长
        num_iter: 迭代次数
        decay: 动量衰减因子
        prob: 应用变换的概率
    """
    x_adv = x
    # 初始化动量
    momentum = torch.zeros_like(x).detach().to(device)
    
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        
        # 应用多样化输入变换
        x_transformed = diverse_input_transform(x_adv, prob)
        
        loss = loss_fn(model(x_transformed), y)
        loss.backward()
        
        # 动量更新
        grad = x_adv.grad.detach()
        momentum = decay * momentum + grad / torch.norm(grad, p=1)
        
        # 更新对抗样本
        x_adv = x_adv + alpha * momentum.sign()
        
        # 裁剪到允许范围
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    
    return x_adv

def diverse_input_transform(x, prob=0.5):
    """
    多样化输入变换函数
    Args:
        x: 输入图像张量，形状为 (batch_size, channels, height, width)
        prob: 应用变换的概率
    Returns:
        变换后的图像张量
    """
    batch_size, channels, height, width = x.shape
    
    # 随机决定是否应用变换
    if torch.rand(1).item() > prob:
        return x
    
    # 随机选择变换类型：0-resize, 1-padding
    transform_type = torch.randint(0, 2, (1,)).item()
    
    if transform_type == 0:
        # Random resizing
        return random_resize(x)
    else:
        # Random padding
        return random_padding(x)

def random_resize(x):
    """
    随机调整图像大小
    Args:
        x: 输入图像张量，形状为 (batch_size, channels, height, width)
    Returns:
        调整大小后的图像张量
    """
    batch_size, channels, height, width = x.shape
    
    # 随机选择新的尺寸（在原尺寸的80%-120%之间）
    resize_ratio = 0.8 + 0.4 * torch.rand(1).item()  # [0.8, 1.2]
    new_size = int(height * resize_ratio)
    
    # 确保新尺寸至少为16，且不超过原尺寸的1.5倍
    new_size = max(16, min(new_size, int(height * 1.5)))
    
    # 使用双线性插值调整大小
    x_resized = torch.nn.functional.interpolate(
        x, size=(new_size, new_size), mode='bilinear', align_corners=False
    )
    
    # 如果图像变小了，需要用零填充回原尺寸
    if new_size < height:
        pad_size = (height - new_size) // 2
        x_resized = torch.nn.functional.pad(
            x_resized, 
            (pad_size, height - new_size - pad_size, pad_size, height - new_size - pad_size),
            mode='constant', value=0
        )
    # 如果图像变大了，需要裁剪回原尺寸
    elif new_size > height:
        start = (new_size - height) // 2
        x_resized = x_resized[:, :, start:start+height, start:start+width]
    
    return x_resized

def random_padding(x):
    """
    随机在图像周围填充零
    Args:
        x: 输入图像张量，形状为 (batch_size, channels, height, width)
    Returns:
        填充后再裁剪回原尺寸的图像张量
    """
    batch_size, channels, height, width = x.shape
    
    # 随机选择填充大小（最多为原尺寸的10%）
    max_pad = height // 10
    pad_left = torch.randint(0, max_pad + 1, (1,)).item()
    pad_right = torch.randint(0, max_pad + 1, (1,)).item()
    pad_top = torch.randint(0, max_pad + 1, (1,)).item()
    pad_bottom = torch.randint(0, max_pad + 1, (1,)).item()
    
    # 应用填充
    x_padded = torch.nn.functional.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0
    )
    
    # 随机裁剪回原尺寸
    padded_height, padded_width = x_padded.shape[2], x_padded.shape[3]
    start_h = torch.randint(0, padded_height - height + 1, (1,)).item()
    start_w = torch.randint(0, padded_width - width + 1, (1,)).item()
    
    x_cropped = x_padded[:, :, start_h:start_h+height, start_w:start_w+width]
    
    return x_cropped

# 定义函数：执行对抗攻击并生成对抗样本
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()  # 设置模型为评估模式
    adv_names = []  # 存储对抗样本名称（实际未使用）
    train_acc, train_loss = 0.0, 0.0  # 初始化准确率和损失累计器
    # 遍历数据加载器中的每个批次
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)  # 将数据移动到设备
        x_adv = attack(model, x, y, loss_fn)  # 使用指定的攻击方法生成对抗样本
        yp = model(x_adv)  # 对对抗样本进行预测
        loss = loss_fn(yp, y)  # 计算损失
        # 累计在对抗样本上的准确预测数量
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        # 累计损失
        train_loss += loss.item() * x.shape[0]
        # 存储对抗样本：需要反标准化和格式转换
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)  # 反标准化到[0,1]范围
        adv_ex = (adv_ex * 255).clamp(0, 255)  # 缩放到[0,255]范围
        adv_ex = adv_ex.detach().cpu().data.numpy().round()  # 转换为numpy数组并四舍五入
        adv_ex = adv_ex.transpose((0, 2, 3, 1))  # 转置维度从(bs,C,H,W)到(bs,H,W,C)
        # 将当前批次的对抗样本添加到总的对抗样本数组中
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    # 返回对抗样本、准确率和损失
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# 定义函数：创建目录并保存对抗样本
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    # 如果对抗样本目录不存在，则复制原始数据目录结构
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    # 遍历每个对抗样本和对应的名称
    for example, name in zip(adv_examples, adv_names):
        # 将numpy数组转换为PIL图像（像素值必须是无符号整数）
        im = Image.fromarray(example.astype(np.uint8))
        # 保存图像到指定路径
        im.save(os.path.join(adv_dir, name))

# 从pytorchcv库导入预训练模型
from pytorchcv.model_provider import get_model as ptcv_get_model

# 加载预训练的ResNet-110模型（在CIFAR-10上训练）并移动到设备
model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 评估模型在良性图像上的性能
benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

# 执行FGSM攻击
adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

# 创建FGSM对抗样本目录并保存图像
create_dir(root, 'fgsm', adv_examples, adv_names)

# 执行I-FGSM攻击
adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

# 创建I-FGSM对抗样本目录并保存图像
create_dir(root, 'ifgsm', adv_examples, adv_names)

# 执行MI-FGSM攻击
adv_examples, mifgsm_acc, mifgsm_loss = gen_adv_examples(model, adv_loader, mifgsm, loss_fn)
print(f'mifgsm_acc = {mifgsm_acc:.5f}, mifgsm_loss = {mifgsm_loss:.5f}')

# 创建MI-FGSM对抗样本目录并保存图像
create_dir(root, 'mifgsm', adv_examples, adv_names)

# 执行DIM-MIFGSM攻击
adv_examples, dim_mifgsm_acc, dim_mifgsm_loss = gen_adv_examples(model, adv_loader, dim_mifgsm, loss_fn)
print(f'dim_mifgsm_acc = {dim_mifgsm_acc:.5f}, dim_mifgsm_loss = {dim_mifgsm_loss:.5f}')

# 创建DIM-MIFGSM对抗样本目录并保存图像
create_dir(root, 'dim_mifgsm', adv_examples, adv_names)

# 集成攻击
class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        # 创建多个预训练模型的模块列表
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # 初始化集成logits为None
        ensemble_logits = None
        for i, m in enumerate(self.models):
            # 获取当前模型的logits输出
            logits = m(x)
            # 如果是第一个模型，直接赋值；否则累加logits
            if ensemble_logits is None:
                ensemble_logits = logits
            else:
                ensemble_logits += logits
        # 返回集成后的平均logits
        return ensemble_logits / len(self.models)

# 构建集成模型
model_names = [
    'nin_cifar10',        # Network in Network
    'resnet20_cifar10',   # ResNet-20
    'preresnet20_cifar10' # Pre-activation ResNet-20
]
ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()

# 使用集成模型执行MI-FGSM攻击
ensemble_mifgsm_examples, ensemble_mifgsm_acc, ensemble_mifgsm_loss = gen_adv_examples(ensemble_model, adv_loader, mifgsm, loss_fn)
print(f'ensemble_mifgsm_acc = {ensemble_mifgsm_acc:.5f}, ensemble_mifgsm_loss = {ensemble_mifgsm_loss:.5f}')

# 创建集成MI-FGSM对抗样本目录并保存图像
create_dir(root, 'ensemble_mifgsm', ensemble_mifgsm_examples, adv_names)

# 使用集成模型执行DIM-MIFGSM攻击
ensemble_dim_mifgsm_examples, ensemble_dim_mifgsm_acc, ensemble_dim_mifgsm_loss = gen_adv_examples(ensemble_model, adv_loader, dim_mifgsm, loss_fn)
print(f'ensemble_dim_mifgsm_acc = {ensemble_dim_mifgsm_acc:.5f}, ensemble_dim_mifgsm_loss = {ensemble_dim_mifgsm_loss:.5f}')

# 创建集成DIM-MIFGSM对抗样本目录并保存图像
create_dir(root, 'ensemble_dim_mifgsm', ensemble_dim_mifgsm_examples, adv_names)

# 可视化部分
import matplotlib.pyplot as plt

# CIFAR-10的类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 创建图形窗口，设置大小为10x20英寸
plt.figure(figsize=(10, 20))
cnt = 0  # 子图计数器
# 遍历每个类别
for i, cls_name in enumerate(classes):
    # 构建图像路径（每个类别的第一张图像）
    path = f'{cls_name}/{cls_name}1.png'
    
    # 显示良性图像
    cnt += 1  # 增加子图计数
    plt.subplot(len(classes), 4, cnt)  # 创建子图：10行4列的第cnt个
    im = Image.open(f'./data/{path}')  # 打开良性图像
    # 对图像进行预测
    logit = model(transform(im).unsqueeze(0).to(device))[0]  # 添加批次维度并预测
    predict = logit.argmax(-1).item()  # 获取预测类别
    prob = logit.softmax(-1)[predict].item()  # 获取预测概率
    # 设置子图标题，显示图像信息和预测结果
    plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')  # 隐藏坐标轴
    plt.imshow(np.array(im))  # 显示图像
    
    # 显示对抗图像
    cnt += 1  # 增加子图计数
    plt.subplot(len(classes), 4, cnt)  # 创建下一个子图
    im = Image.open(f'./ensemble_dim_mifgsm/{path}')
    logit = ensemble_model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'Ensemble DIM-MIFGSM\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示图形