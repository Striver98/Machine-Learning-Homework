# 导入所需的包
import json  # 用于读取和解析json文件
import numpy as np  # 用于数值计算
import random  # 用于随机数生成
import torch  # PyTorch主库
from torch.utils.data import DataLoader, Dataset  # 数据加载和自定义数据集
from transformers import BertForQuestionAnswering, BertTokenizerFast  # BERT问答模型和分词器
from torch.optim import AdamW  # AdamW优化器

from tqdm.auto import tqdm  # 进度条显示

from torch.optim.lr_scheduler import LambdaLR # 学习率调度器

# 设置设备为cuda（GPU）或cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# 固定随机种子，保证实验可复现
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(20030908)

# 是否开启混合精度训练（fp16），默认不开启
fp16_training = True  # 设置为True开启混合精度训练

if fp16_training:
    from accelerate import Accelerator  # accelerate库用于加速训练
    accelerator = Accelerator(fp16=True)
    device = accelerator.device  # 使用accelerate管理的设备

# 加载BERT问答模型和分词器（中文）
model = BertForQuestionAnswering.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large").to(device)
tokenizer = BertTokenizerFast.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")

# 读取数据函数，返回问题和段落
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

# 读取训练、验证、测试集
train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

# 对问题和段落分别进行分词（不加特殊符号，后续拼接时再加）
train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# 自定义问答数据集
class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split  # 数据集类型（train/dev/test）
        self.questions = questions  # 问题列表
        self.tokenized_questions = tokenized_questions  # 分词后的问题
        self.tokenized_paragraphs = tokenized_paragraphs  # 分词后的段落
        self.max_question_len = 40  # 问题最大长度
        self.max_paragraph_len = 150  # 段落最大长度
        self.doc_stride = 75  # 滑动窗口步长
        # 输入序列最大长度：[CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)  # 返回样本数

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        # 训练集处理
        if self.split == "train":
            # 将答案的字符起止位置转换为分词后的token位置
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
            L = len(tokenized_paragraph)
            min_start = max(0, answer_end_token - self.max_paragraph_len + 1)
            max_start = min(answer_start_token, L - self.max_paragraph_len)
            if max_start < min_start:
                paragraph_start = min_start  # 兜底，窗口只能唯一确定
            else:
                paragraph_start = random.randint(min_start, max_start)
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # 拼接问题和段落，加上特殊符号
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # 答案token位置映射到窗口内
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # 填充并返回模型输入
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # 验证/测试集处理
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            # 段落滑窗，生成多个窗口
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            # 返回所有窗口的输入
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # 计算需要填充的长度
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # 拼接输入id
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # 生成token_type_ids，问题为0，段落为1，pad为0
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # 生成attention_mask，pad为0，其余为1
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

# 构建数据集对象
train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 8  # 训练批次大小

# 构建数据加载器
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# 评估函数，输出预测答案
def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]  # 窗口数量
    for k in range(num_of_windows):
        # 取每个窗口中概率最大的起止位置
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        prob = start_prob + end_prob  # 概率和
        if prob > max_prob:
            max_prob = prob
            # 解码token为字符串
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    # 去除答案中的空格
    return answer.replace(' ','') 

# 训练参数设置
num_epoch = 1  # 训练轮数
validation = True  # 是否验证
logging_step = 100  # 日志打印步数
learning_rate = 1e-4  # 学习率
optimizer = AdamW(model.parameters(), lr=learning_rate)  # 优化器

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

# 学习率调度器，线性衰减
# 计算总步数
total_steps = len(train_loader) * num_epoch
# 学习率衰减函数，线性衰减到0
lr_lambda = lambda step: max(0.0, 1.0 - step / total_steps)
# 创建学习率调度器
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


model.train()  # 设置为训练模式

print("Start Training ...")

# 训练主循环
for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    for data in tqdm(train_loader):	
        # 数据转到GPU
        data = [i.to(device) for i in data]
        # 前向传播，计算损失
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        # 计算预测的起止位置
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        # 统计准确率
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        # 反向传播
        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # TODO: 可加学习率衰减
        scheduler.step()


        # 每logging_step步打印一次训练损失和准确率
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    # 验证集评估
    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # 预测答案与真实答案完全一致才算正确
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# # 保存模型和配置文件到saved_model目录
# print("Saving Model ...")
# model_save_dir = "saved_model" 
# model.save_pretrained(model_save_dir)

# 测试集推理
print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output))

# 写入结果到result.csv
result_file = "result.csv"
with open(result_file, 'w', encoding="utf-8") as f:  # 添加 encoding="utf-8"
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # 答案中的逗号去掉，避免csv格式问题
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")