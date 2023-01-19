# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'simcse-roberta-lstm'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.roberta_path1 = './simcse-roberta-la'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.roberta_path1)
        self.hidden_size = 1024
        self.dropout = 0.2
        self.rnn_hidden = 1024
        self.num_layers = 2

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.roberta_path1)
        for param in self.roberta.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rnn_hidden,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        # self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)

    def forward(self, x):
        #上游任务
        context = x[0]  # 输入的句子 shape[32,64]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]shape[32,64]
        bert_output = self.roberta(context, attention_mask=mask,output_hidden_states=True)#[32,768]
        output=bert_output.last_hidden_state       #最后一层的隐藏状态

        #下游任务
        out, _ = self.lstm(output)
        # out = self.dropout(out)
        out= out[:, -1, :]
        out = self.fc(out)  # 句子最后时刻的 hidden state

        return out
