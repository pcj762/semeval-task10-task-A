import time
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import build_iterator, get_time_dif, build_dataset


def submit(config, model):

    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    data = []
    PAD, CLS = '[PAD]', '[CLS]'
    pad_size = 64
    test_iter = []

    test = pd.read_csv('./data/dev_task_a_entries.csv')
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
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
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
    df.to_csv(r"./data/dev_task_a_submit_"+config.model_name+".csv", index=False)

if __name__ == '__main__':
    dataset = 'data'  # 数据集
    model_name = 'simcse-roberta-lstm'  # 模型名字
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    submit(config, model)