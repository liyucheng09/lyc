from torch.utils.data import (DataLoader,
                              TensorDataset,
                              Dataset,
                              IterableDataset)
from transformers import (BertModel,
                          BertTokenizer,
                          AutoModel,
                          AutoTokenizer)
from datasets import load_dataset, concatenate_datasets, Dataset as hfds
from tqdm import tqdm
import numpy as np
import torch
import os
import pickle

def get_tokenized_ds(scripts, path, tokenizer, max_length=64, slice=None, num_proc=None, shuffle=False, tokenize_type='general'):
    """
    Given huggingface dataset-loading scripts and datapath, return processed datasets.

    Args:
        scripts: .py file
        path: datapath corresponding to the scripts
        tokenizer: huggingface tokenizer object
        ds: the name of the datasets
        slice: if given, will return a slice of the process dataset for testing usage.

    Returns:
        ds: python dict object, where features names are the key, values are pytorch tensor.
    """

    def _slice(ds, slice):
        for k,v in ds.items():
            ds[k]=hfds.from_dict(v[:slice])
        return ds

    def _tokenize1(ds):
        """
        This function return tokenized results nestedly. 返回了嵌套结构，不能使用多进程
        Return:
            {'texta':{'input_ids': ..., 'attention_mask': ...}, ...}
        """
        results={}
        for k,v in ds.items():
            if k == 'label':
                results[k]=v
                continue
            out_=tokenizer(v, max_length=max_length, padding=True, truncation=True)
            results[k]=dict(out_)
        return results

    def _tokenize2(ds):
        """
        本方法返回非嵌套结构，可以使用多进程处理。返回的features中的token_ids前面会加上features_name
        """
        results={}
        for k,v in ds.items():
            if k == 'label':
                results.update({k:v})
                continue
            out_=tokenizer(v, max_length=max_length, padding=True, truncation=True)
            out_={k+'-'+k_: v_ for k_, v_ in out_.items()}
            results.update(out_)
        return results
    
    def _tokenize3(ds):
        results={}
        for k,v in ds.items():
            if k == 'label':
                results[k]=v
                continue
            out_=tokenizer(v, max_length=max_length, padding=True, truncation=True)
            results.update(out_)
        return results

    tokenize_funcs={
        'nested': _tokenize1,
        'with_prefix': _tokenize2,
        'general': _tokenize3
    }
    
    def _get_col_names(col_names):
        if isinstance(col_names, list):
            return col_names
        elif isinstance(col_names, dict):
            cols_needed_removed=set()
            for k,v in col_names.items():
                cols_needed_removed.update(v)
            return cols_needed_removed

    ds=load_dataset(scripts, data_path=path)

    # if ds_name in ds.column_names.keys():
    #     ds=ds[ds_name]
    # elif 'train' in ds.column_names.keys():
    #     train_ds = ds['train']
    # elif 'dev' in ds.column_names.keys():
    #     dev_ds = ds['dev']
    # elif 'test' in ds.column_names.keys():
    #     test_ds = ds['test']

    ds = _slice(ds, slice) if slice else ds

    cols_needed_removed=_get_col_names(ds.column_names)

    ds = ds.map(
        tokenize_funcs[tokenize_type],
        remove_columns=cols_needed_removed,
        batched=True,
        num_proc=num_proc
    )

    if shuffle:
        ds=ds.shuffle()
    
    return ds

def get_dataloader(ds: hfds, batch_size=32, cols=['input_ids', 'attention_mask', 'token_type_ids', 'label']):
    ds.set_format(type='torch', columns=cols)
    dl=DataLoader(ds, batch_size)
    return dl

class SentencePairDataset(Dataset):
    """
    句子对/单句数据集。用于训练/使用句向量模型。

    Train: 
        given sentence pair dataset: (sentence_a, sentence_b, label)
    
    predict:
        given single sentence: (sentence_a)
    """
    def __init__(self, tokenized_a, tokenized_b=None, label=None):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b
        self.label=label

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        output=(input_a, )
        if self.tokenized_b is not None:
            input_b={
                k:v[index] for k,v in self.tokenized_b.items()
            }
            output+=(input_b, )
        if self.label is not None:
            output+=(torch.LongTensor([self.label[index]]), )
        return output

class SimCSEDataSet(IterableDataset):
    def __init__(self, tokenized_a, batch_size=32):
        self.tokenized_a=tokenized_a
        self.batch_size=batch_size
        self.idxs=np.random.permutation(len(self.tokenized_a['input_ids']))

    def __iter__(self):
        count=0
        while count<len(self.idxs):
            selected_ids=self.idxs[count:count+self.batch_size]
            inputs={k: v[selected_ids].repeat(2,1) for k,v in self.tokenized_a.items() if k!='label'}
            idx1=torch.arange(self.batch_size*2)[None, :]
            idx2=(idx1.T+self.batch_size)%(self.batch_size*2)
            label = torch.LongTensor(idx2)
            yield {'inputs': inputs, 'label': label}
            count+=self.batch_size

if __name__ == '__main__':
    from utils import get_tokenizer
    from copy import deepcopy

    t=get_tokenizer('bert-base-chinese', is_zh=True)
    ds = get_tokenized_ds('hfds_scripts/atec_dataset.py', '../sentence-embedding/data/ATEC/atec_nlp_sim_train.csv', t, tokenize_type='with_prefix')

    ds = ds['atec']
    ds2=deepcopy(ds)

    for index, ds_ in enumerate([ds, ds2]):
        features=list(ds_.features)
        for feature in features:
            if index:
                if feature.startswith('textb') or feature == 'label':
                    ds_.remove_columns_(feature)
                else:
                    ds_.rename_column_(feature, feature[6:])
            else:
                if feature.startswith('texta') or feature == 'label':
                    ds_.remove_columns_(feature)
                else:
                    ds_.rename_column_(feature, feature[6:])
    
    ds=concatenate_datasets([ds, ds2])
    print(ds)