# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'roberta_lstm_gru'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 6                                             # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.roberta_path = './roberta-large'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.roberta_path)
        self.hidden_size = 1024
        self.dropout = 0.2
        self.lstm_hidden = 1024
        self.num_layers = 2

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.roberta_path)
        for param in self.roberta.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size,
                            config.lstm_hidden,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.gru = nn.GRU(config.lstm_hidden*2,
                            config.lstm_hidden,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.lstm_hidden*5, config.num_classes)

    def forward(self, x):

        context = x[0]  # 输入的句子 shape[32,64]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]shape[32,64]
        outputs = self.roberta(context, attention_mask=mask,output_hidden_states=True)

        bert_output = outputs[0]
        pooled_output = outputs[1]

#         print(bert_output.shape, pooled_output.shape)

        h_lstm, _ = self.lstm(bert_output)  # [bs, seq, output*dir]
        h_gru, _ = self.gru(h_lstm)
#         print(h_lstm.shape,h_gru.shape,hh_gru.shape)
#         hh_gru = hh_gru.view(-1,2*1024)

#         print(h_lstm.shape,h_gru.shape,hh_gru.shape)
        avg_pool = torch.mean(h_gru, 1)   #进行池化操作
        max_pool, _ = torch.max(h_gru, 1)
#       print( avg_pool.shape, hh_gru.shape, max_pool.shape, pooled_output.shape)

        h_conc_a = torch.cat(   #进行拼接
            (avg_pool, max_pool, pooled_output), 1
        )
#         print(h_conc_a.shape)
        output = self.dropout(h_conc_a)
        out = self.fc(output)

        return out

