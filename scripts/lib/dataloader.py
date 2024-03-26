import pandas as pd
from sklearn.metrics import accuracy_score

data_dir = {"mmlu": "./data/benchmark/mmlu/mmlu_mingqian.csv", 
            "arc": "./data/benchmark/arc/ARC-Challenge-Test.csv"}

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

class benchmark_mmlu:
    def __init__(self):
        self.name = "mmlu"
        self.data_df = pd.read_csv(data_dir[self.name])
        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['question'].strip()}\nA. {item['option1']}\nB. {item['option2']}\nC. {item['option3']}\nD. {item['option4']}\n"
            self.question_list.append(q_text)
        self.true_label_list = self.data_df["true_option"]
    
    def load_question_list(self):
        return self.question_list

    def eval_question_list(self, pred_text_list, vllm=True):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            text = pred_text.outputs[0].text.replace("\n", "").strip()
            model_choice_tmp = "Z"
            for c in text:
                if c in ["A", "B", "C", "D"]:
                    model_choice_tmp = c
                    break
            pred_label_list.append(letter2num[model_choice_tmp])
            if model_choice_tmp == "Z":
                errors += 1
        return accuracy_score(self.true_label_list, pred_label_list), error_num

class benchmark_arc:
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
    
    def load_question_list(self):
        return self.question_list

    def eval_question_list(self, pred_text_list, vllm=True):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            text = pred_text.outputs[0].text.replace("\n", "").strip()
            model_choice_tmp = "Z"
            for c in text:
                if c in ["A", "B", "C", "D"]:
                    model_choice_tmp = c
                    break
            pred_label_list.append(model_choice_tmp)
            if model_choice_tmp == "Z":
                error_num += 1
        return accuracy_score(self.true_label_list, pred_label_list), error_num

def init_benchmark(name="mmlu"):
    if name == "mmlu":
        return benchmark_mmlu()
    elif name == "arc":
        return benchmark_arc()