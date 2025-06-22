# ��������İ�
import json  # ���ڶ�ȡ�ͽ���json�ļ�
import numpy as np  # ������ֵ����
import random  # �������������
import torch  # PyTorch����
from torch.utils.data import DataLoader, Dataset  # ���ݼ��غ��Զ������ݼ�
from transformers import BertForQuestionAnswering, BertTokenizerFast  # BERT�ʴ�ģ�ͺͷִ���
from torch.optim import AdamW  # AdamW�Ż���

from tqdm.auto import tqdm  # ��������ʾ

from torch.optim.lr_scheduler import LambdaLR # ѧϰ�ʵ�����

# �����豸Ϊcuda��GPU����cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# �̶�������ӣ���֤ʵ��ɸ���
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

# �Ƿ�����Ͼ���ѵ����fp16����Ĭ�ϲ�����
fp16_training = True  # ����ΪTrue������Ͼ���ѵ��

if fp16_training:
    from accelerate import Accelerator  # accelerate�����ڼ���ѵ��
    accelerator = Accelerator(fp16=True)
    device = accelerator.device  # ʹ��accelerate������豸

# ����BERT�ʴ�ģ�ͺͷִ��������ģ�
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# ��ȡ���ݺ�������������Ͷ���
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

# ��ȡѵ������֤�����Լ�
train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

# ������Ͷ���ֱ���зִʣ�����������ţ�����ƴ��ʱ�ټӣ�
train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

# �Զ����ʴ����ݼ�
class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split  # ���ݼ����ͣ�train/dev/test��
        self.questions = questions  # �����б�
        self.tokenized_questions = tokenized_questions  # �ִʺ������
        self.tokenized_paragraphs = tokenized_paragraphs  # �ִʺ�Ķ���
        self.max_question_len = 40  # ������󳤶�
        self.max_paragraph_len = 150  # ������󳤶�
        self.doc_stride = 75  # �������ڲ���
        # ����������󳤶ȣ�[CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)  # ����������

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        # ѵ��������
        if self.split == "train":
            # ���𰸵��ַ���ֹλ��ת��Ϊ�ִʺ��tokenλ��
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # �Դ�Ϊ���Ľ�ȡһ�δ���
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # ƴ������Ͷ��䣬�����������
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # ��tokenλ��ӳ�䵽������
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # ��䲢����ģ������
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # ��֤/���Լ�����
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            # ���们�������ɶ������
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            # �������д��ڵ�����
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # ������Ҫ���ĳ���
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # ƴ������id
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # ����token_type_ids������Ϊ0������Ϊ1��padΪ0
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # ����attention_mask��padΪ0������Ϊ1
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

# �������ݼ�����
train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 32  # ѵ�����δ�С

# �������ݼ�����
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# �������������Ԥ���
def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]  # ��������
    for k in range(num_of_windows):
        # ȡÿ�������и���������ֹλ��
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        prob = start_prob + end_prob  # ���ʺ�
        if prob > max_prob:
            max_prob = prob
            # ����tokenΪ�ַ���
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    # ȥ�����еĿո�
    return answer.replace(' ','') 

# ѵ����������
num_epoch = 1  # ѵ������
validation = True  # �Ƿ���֤
logging_step = 100  # ��־��ӡ����
learning_rate = 1e-4  # ѧϰ��
optimizer = AdamW(model.parameters(), lr=learning_rate)  # �Ż���

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

# ѧϰ�ʵ�����������˥��
# �����ܲ���
total_steps = len(train_loader) * num_epoch
# ѧϰ��˥������������˥����0
lr_lambda = lambda step: max(0.0, 1.0 - step / total_steps)
# ����ѧϰ�ʵ�����
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


model.train()  # ����Ϊѵ��ģʽ

print("Start Training ...")

# ѵ����ѭ��
for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    for data in tqdm(train_loader):	
        # ����ת��GPU
        data = [i.to(device) for i in data]
        # ǰ�򴫲���������ʧ
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        # ����Ԥ�����ֹλ��
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        # ͳ��׼ȷ��
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        # ���򴫲�
        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # TODO: �ɼ�ѧϰ��˥��
        scheduler.step()


        # ÿlogging_step����ӡһ��ѵ����ʧ��׼ȷ��
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    # ��֤������
    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # Ԥ�������ʵ����ȫһ�²�����ȷ
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# # ����ģ�ͺ������ļ���saved_modelĿ¼
# print("Saving Model ...")
# model_save_dir = "saved_model" 
# model.save_pretrained(model_save_dir)

# ���Լ�����
print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output))

# д������result.csv
result_file = "result.csv"
with open(result_file, 'w', encoding="utf-8") as f:  # ��� encoding="utf-8"
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # ���еĶ���ȥ��������csv��ʽ����
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")