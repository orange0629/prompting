import pandas as pd
from sklearn.metrics import accuracy_score
import os
import re

data_dir = {"mmlu": "./data/benchmark/mmlu/mmlu_mingqian.csv", 
            "arc": "./data/benchmark/arc/ARC-Challenge-Test.csv",
            "hellaswag": "./data/benchmark/hellaswag/hellaswag_train.jsonl"}
save_intermediate_dir = "./results/benchmark"

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

class benchmark_base:
    def __init__(self):
        self.name = "base"
        self.data_df, self.question_list, self.true_label_list = pd.DataFrame(), [], []

    def get_choice_from_text(self, text) -> str:
        model_choice_tmp = "Z"
        for c in text:
            if c in ["A", "B", "C", "D"]:
                model_choice_tmp = c
                break
        return model_choice_tmp
    
    def save_intermediate(self, pred_label_list, model_name, column_name):
        if not os.path.exists(save_intermediate_dir):
            os.makedirs(save_intermediate_dir)
        save_dir_tmp = f"{save_intermediate_dir}/{model_name}_{self.name}_results.csv"
        try:
            save_df = pd.read_csv(save_dir_tmp)
        except:
            save_df = self.data_df.copy()
        save_df[column_name] = pred_label_list
        save_df.to_csv(save_dir_tmp, index=False)
    
    def clean_text(self, text):
        pattern = r'[^a-zA-Z0-9 !#$%&()*+,.:;<=>?@_{|}-]'
        cleaned_text = re.sub(pattern, ' ', text)
        return re.sub("\s\s+" , " ", cleaned_text).strip()
    
    def load_question_list(self):
        return self.question_list

    def eval_question_list(self, pred_text_list, vllm=True, save_intermediate=(False, "", "")):
        return dict()

class benchmark_mmlu(benchmark_base):
    def __init__(self):
        self.name = "mmlu"
        self.data_df = pd.read_csv(data_dir[self.name])

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['question'].strip()}\nA. {item['option1']}\nB. {item['option2']}\nC. {item['option3']}\nD. {item['option4']}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["true_option"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = num2letter[self.true_label_list[idx]]

    def eval_question_list(self, pred_text_list, vllm=True, save_intermediate=(False, "", "")):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            if vllm:
                text = self.clean_text(pred_text.outputs[0].text)
            else:
                text = self.clean_text(pred_text)
            model_choice_tmp = self.get_choice_from_text(text)
            if model_choice_tmp == "Z":
                pred_label_list.append(text)
                error_num += 1
            else:
                pred_label_list.append(model_choice_tmp)
        
        if save_intermediate[0]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])

        metrics = {f"{self.name.upper()}_acc": accuracy_score(self.true_label_list, pred_label_list),
                   f"{self.name.upper()}_acc_no_error": (accuracy_score(self.true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num),
                   f"{self.name.upper()}_error": error_num}

        return metrics

class benchmark_arc(benchmark_base):
    def __init__(self):
        self.name = "arc"
        self.data_df = pd.read_csv(data_dir[self.name])

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = item["question"].strip().replace("(1)", "(A)").replace("(2)", "(B)").replace("(3)", "(C)").replace("(4)", "(D)") + "\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["AnswerKey"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = self.true_label_list[idx].upper().strip()
            if self.true_label_list[idx] in num2letter:
                self.true_label_list[idx] = num2letter[self.true_label_list[idx]]
    

    def eval_question_list(self, pred_text_list, vllm=True, save_intermediate=(False, "", "")):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            if vllm:
                text = self.clean_text(pred_text.outputs[0].text)
            else:
                text = self.clean_text(pred_text)
            model_choice_tmp = self.get_choice_from_text(text)
            if model_choice_tmp == "Z":
                pred_label_list.append(text)
                error_num += 1
            else:
                pred_label_list.append(model_choice_tmp)
        
        if save_intermediate[0]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {f"{self.name.upper()}_acc": accuracy_score(self.true_label_list, pred_label_list),
                   f"{self.name.upper()}_acc_no_error": (accuracy_score(self.true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num),
                   f"{self.name.upper()}_error": error_num}

        return metrics

class benchmark_hellaswag(benchmark_base):
    def __init__(self):
        self.name = "hellaswag"
        self.data_df = pd.read_json(path_or_buf=data_dir[self.name], lines=True).sample(n=1000, random_state=42)

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['ctx'].strip()}\nA. {item['endings'][0]}\nB. {item['endings'][1]}\nC. {item['endings'][2]}\nD. {item['endings'][3]}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["label"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = num2letter[int(self.true_label_list[idx])+1]


    def eval_question_list(self, pred_text_list, vllm=True, save_intermediate=(False, "", "")):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            if vllm:
                text = self.clean_text(pred_text.outputs[0].text)
            else:
                text = self.clean_text(pred_text)
            model_choice_tmp = self.get_choice_from_text(text)
            if model_choice_tmp == "Z":
                pred_label_list.append(text)
                error_num += 1
            else:
                pred_label_list.append(model_choice_tmp)
        
        if save_intermediate[0]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])

        metrics = {f"{self.name.upper()}_acc": accuracy_score(self.true_label_list, pred_label_list),
                   f"{self.name.upper()}_acc_no_error": (accuracy_score(self.true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num),
                   f"{self.name.upper()}_error": error_num}

        return metrics

def init_benchmark(name="mmlu") -> benchmark_base:
    if name == "mmlu":
        return benchmark_mmlu()
    elif name == "arc":
        return benchmark_arc()
    elif name == "hellaswag":
        return benchmark_hellaswag()