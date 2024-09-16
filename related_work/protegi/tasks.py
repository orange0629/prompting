import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
from lib.dataloader import init_benchmark

import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass




def process_example(ex, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class EthosBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[200:]]
        return exs
    
    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[:200]]
        return exs


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs


def process_example_vllm(ex_lst, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class CustomTask(DataProcessor):
    def __init__(self, benchmark_name):
        self.benchmark_obj = init_benchmark(benchmark_name)

    def evaluate(self, predictor, prompt, test_exs, n=100):
        preds = predictor.inference(test_exs[:n], prompt)
        #score, texts, labels, preds = process_example_vllm(test_exs[:n], predictor, prompt)
        texts = [ex['text'] for ex in test_exs[:n]]
        labels = [ex['label'] for ex in test_exs[:n]]
        test_idx = [ex['id'] for ex in test_exs[:n]]
        metric = self.benchmark_obj.eval_question_list(preds, ("eval", "", ""), test_idx, return_error_idx=True)
        score = metric[list(metric.keys())[0]]
        error_idx = metric[list(metric.keys())[-1]]

        return score, texts, labels, preds, error_idx
    
    def get_train_examples(self):
        qlist, idx_list = self.benchmark_obj.load_random_question_list(num_q=None, split="train")
        label_full = self.benchmark_obj.true_label_list
        exs = []
        for idx in range(len(qlist)):
            exs.append({"id": idx_list[idx], "text": qlist[idx], "label": label_full[idx_list[idx]]})
        return exs
    
    def get_test_examples(self):
        qlist, idx_list = self.benchmark_obj.load_random_question_list(num_q=None, split="test")
        label_full = self.benchmark_obj.true_label_list
        exs = []
        for idx in range(len(qlist)):
            exs.append({"id": idx_list[idx], "text": qlist[idx], "label": label_full[idx_list[idx]]})
        return exs
    
    def stringify_prediction(self, pred):
        return str(pred)
    
