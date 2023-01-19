from importlib import import_module
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
from utils import build_dataset, build_iterator


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bagging'
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
        self.roberta_path = './roberta-large'
        self.bert_path = './ERNIE_pretrain'
        self.roberta_path1 = './simcse-roberta-la'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.roberta_path)
        
        self.hidden_size = 1024
        self.dropout = 0.2
        self.rnn_hidden = 1024
        self.num_layers = 2
        self.lstm_hidden = 1024
#         self.hidden_size1 = 768

def predict_single(outputs):
    preds = torch.argmax(outputs,dim=1) #每一行的最大值下标
    return preds

class Bagging_models():
    def __init__(self,model_list,model_name,config):
        self.model_list = model_list
        self.model_name= model_name
        self.models = []
        for name,modelname in zip(model_list , model_name):
            x = import_module('models.' + modelname)
            model = x.Model(config).to(config.device)
            model.load_state_dict(torch.load(name),False)
            self.models.append(model)

    def __call__(self, inputs):
        outputs = [model(inputs)
                   for model in self.models]
        preds = np.array([predict_single(o).cpu().numpy() for o in outputs])
        bagging_result = []
        for i in range(preds.shape[1]):
            bagging_result.append(np.argmax(np.bincount(preds[:,i])))

        return torch.from_numpy(np.array(bagging_result)).to(config.device)


def test(model, data_loader, config):
    hit = 0
    total = 0
    res_preds = []
    res_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            inputs, labels = data
            outputs = model(inputs)
            preds = outputs

            res_labels.extend(list(labels.cpu().numpy()))
            res_preds.extend(list(preds.cpu().numpy()))

            hit += sum(labels == preds).item()
            total += len(labels)

        acc = hit/total     #准确率

        label_class = list(set(res_labels))
        P = []
        R = []
        res_labels = np.array(res_labels)
        res_preds = np.array(res_preds)
        for c in label_class:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(res_labels):
                p = res_preds[i]
                if(p==c and r == c):
                    tp += 1
                elif(p == c and r != c ):
                    fp += 1
                elif( p != c and r != c):
                    tn += 1
                elif( p != c and r == c):
                    fn += 1
            P.append(tp/(tp + fp))
            R.append(tp/(tp + fn))
        Macro_P = np.mean(P)
        Macro_R = np.mean(R)
        Macro_F = 2 * Macro_P*Macro_R / (Macro_P + Macro_R)

        print("准确率: {}, 平均精确率: {}, 平均召回率: {}, 平均F值: {}"
              .format(acc, Macro_R, Macro_R, Macro_F))
def submit(config, model):
    # test
    data = []
    PAD, CLS = '[PAD]', '[CLS]'
    pad_size = 64
    test_iter = []

    test = pd.read_csv('./data/test_task_a_entries.csv')
    test_text = test['text'].values.tolist()
    test_id = test['rewire_id'].values.tolist()

    for line in tqdm(range(0, len(test_text))):
        content = test_text[line]
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        test_iter.append((token_ids, int(1), seq_len, mask))
    test_iter = build_iterator(test_iter, config)

    with torch.no_grad():
        predict_all = np.array([], dtype=int)
        for texts, labels in test_iter:
            outputs= model(texts)
            predic = outputs.cpu().numpy()
            predict_all = np.append(predict_all, predic)

    dir = {
        0: "not sexist",
        1: "sexist"
    }
    predict_all = [dir[n] for n in predict_all]
    data = {
        'rewire_id': test_id,
        'label_pred': predict_all
    }
    df = pd.DataFrame(data)
    df.to_csv(r"./data/test_task_a_submit.csv", index=False)
    
def test_ignore_cluster(model, data_loader, args):
    res_preds = []
    res_labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

            inputs, labels= data
            outputs = model(inputs)
            preds = outputs

            res_labels.extend(list(labels.cpu().numpy()))
            res_preds.extend(list(preds.cpu().numpy()))

        label_class = [0,1]
        P = []
        R = []
        acc = 0
        res_labels = np.array(res_labels)
        res_preds = np.array(res_preds)
        for c in label_class:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(res_labels):
                p = res_preds[i]
                if p == 1 or p == 2 or p == 3 or p == 4:
                    p = 1
                if r == 1 or r == 2 or r == 3 or r == 4:
                    r = 1
                if(p==c and r == c):
                    tp += 1
                elif(p == c and r != c ):
                    fp += 1
                elif( p != c and r != c):
                    tn += 1
                elif( p != c and r == c):
                    fn += 1

            acc += (tp + tn)/(tp + fn + fp + tn)

            if tp != 0:
                P.append(tp/(tp + fp))
                R.append(tp/(tp + fn))
            else:
                P.append(0)
                R.append(0)

        Macro_P = np.mean(P)
        Macro_R = np.mean(R)
        Macro_F = 2 * Macro_P*Macro_R / (Macro_P + Macro_R)
        acc = acc / 3
        print("准确率: {}, 平均精确率: {}, 平均召回率: {}, 平均F值: {}"
              .format(acc, Macro_P, Macro_R, Macro_F))
if __name__ == '__main__':
    model_list = ["data/saved_dict/roberta.ckpt", "data/saved_dict/roberta-RNN.ckpt",
                  "data/saved_dict/simcse-roberta-lstm.ckpt", "data/saved_dict/roberta_lstm_gru.ckpt"]
    model_name =['roberta',"roberta-RNN","simcse-roberta-lstm","roberta_lstm_gru"]
    # dataset = 'data'  # 数据集
    config=Config('data')
    model = Bagging_models(model_list, model_name, config)
#     train_data, dev_data, test_data = build_dataset(config)
#     test_iter = build_iterator(test_data, config)
    # test_ignore_cluster(model, test_iter, config)
#     test(model, test_iter, config)
    submit(config,model)