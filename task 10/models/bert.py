# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 1024
        self.dropout=0.2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)    
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子 shape[32,64]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]shape[32,64]
        # output = self.roberta(context, attention_mask=mask)
        output = self.bert(context, attention_mask=mask,output_hidden_states=True)#[32,768]
#         # pooled = output[2][10:13][:, 0, :]
#         # pooled = output[2][9:13]
#         # pooled_mean=(pooled[0][:, 0, :]+pooled[1][:, 0, :]+pooled[2][:, 0, :]+pooled[3][:, 0, :])/4
#         # pooled_mean=self.dropout_bertout(pooled_mean)


#         # pooled = output[1]

        # out = self.fc(pooled_mean) #shape [32,10]
        hidden_states=output[2]
        nopooled_output = torch.cat((hidden_states[9], hidden_states[10], hidden_states[11], hidden_states[12]), 1)
        batch_size = nopooled_output.shape[0]
        kernel_hight = nopooled_output.shape[1]
        pooled_output = F.max_pool2d(nopooled_output, kernel_size=(kernel_hight, 1))
        flatten = pooled_output.view(batch_size, -1)
        flattened_output = self.dropout(flatten)
        out = self.fc(flattened_output)
        return out
