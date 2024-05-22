import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_metric, load_dataset
from tqdm import tqdm
import lib.utils
import os
import re

data_dir = {"mmlu": "./data/benchmark/mmlu/mmlu_mingqian.csv", 
            "arc": "./data/benchmark/arc/ARC-Challenge-Test.csv",
            "hellaswag": "./data/benchmark/hellaswag/hellaswag_train.jsonl",
            "truthfulqa": "./data/benchmark/truthfulqa/TruthfulQA.csv"}
save_intermediate_dir = "./results/benchmark"

MULTIPLE_CHOICE_DEFAULT_USER_PROMPT = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"
MULTIPLE_CHOICE_COT_USER_PROMPT = "The following is a multiple choice question (with answers). Think carefully step by step. Describe your reasoning in steps and then output the option letter at the very end.\n{question_prompt}"

YES_NO_POSTFIX = " Reply with only yes or no."
YES_NO_COT_POSTFIX = " Think carefully step by step. Describe your reasoning in steps and then output yes or no at the very end."

QA_DEFAULT_USER_PROMPT = "{question_prompt}"

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

class benchmark_base:
    def __init__(self):
        self.name = "base"
        self.data_df, self.question_list, self.true_label_list = pd.DataFrame(), [], []
    
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
        pattern = r"[^a-zA-Z0-9 !#$%&()*'\"+,.:;<=>?@_{|}-]"
        cleaned_text = re.sub(pattern, ' ', text)
        return re.sub("\s\s+" , " ", cleaned_text).strip()

    def result_list_preprocessing(self, pred_text_list, args, result_type="multiple_choice"):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            text = self.clean_text(pred_text.outputs[0].text) if not args.hf else self.clean_text(pred_text)
            
            if result_type == "multiple_choice":
                pattern = re.compile(r'[ABCD]')
                matches = list(pattern.finditer(text))
                if matches:
                    if args.cot != 0:
                        pred_label_list.append(matches[-1].group())
                    else:
                        pred_label_list.append(matches[0].group())
                else:
                    pred_label_list.append(text)
                    error_num += 1
            elif result_type == "yes_no":
                pattern = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
                matches = list(pattern.finditer(text))
                if matches:
                    if args.cot != 0:
                        pred_label_list.append(int(matches[-1].group() == "yes"))
                    else:
                        pred_label_list.append(int(matches[0].group() == "yes"))
                else:
                    pred_label_list.append(text)
                    error_num += 1
            else:
                pred_label_list.append(text)

        return pred_label_list, error_num
    
    def load_question_list(self):
        return self.question_list

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        return dict()
    
    def get_user_prompt(self, args):
        if args.cot == 1:
            return MULTIPLE_CHOICE_COT_USER_PROMPT
        else:
            return MULTIPLE_CHOICE_DEFAULT_USER_PROMPT

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

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, args, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
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
    

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, args, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
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


    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, args, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            metrics = {f"{self.name.upper()}_acc": accuracy_score(self.true_label_list, pred_label_list),
                    f"{self.name.upper()}_acc_no_error": (accuracy_score(self.true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num),
                    f"{self.name.upper()}_error": error_num}

        return metrics

class benchmark_truthfulqa(benchmark_base):
    def __init__(self):
        self.name = "truthfulqa"
        self.data_df = pd.read_csv(data_dir[self.name])

        self.question_list = self.data_df["Question"]
        self.true_label_list = list(self.data_df["Best Answer"])

        self.correct_answer_list = [lib.utils.split_multi_answer(text, add_no_comment=True) for text in self.data_df["Correct Answers"]]
        self.incorrect_answer_list = [lib.utils.split_multi_answer(text) for text in self.data_df["Incorrect Answers"]]

        self.bleurt = None
    
    def get_user_prompt(self, args):
        return QA_DEFAULT_USER_PROMPT

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            #if self.bleurt is None:
            #    self.bleurt = load_metric("bleurt")
            #bleurt_tmp = lib.utils.bleurt_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list, self.bleurt)
            bleu_tmp = lib.utils.bleu_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list)
            #rouge_tmp = lib.utils.rouge_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list)

            metrics = {#f"{self.name.upper()}_BLEURT_acc": bleurt_tmp["BLEURT_acc"],
                    f"{self.name.upper()}_BLEU_acc": bleu_tmp["BLEU_acc"],
                    #f"{self.name.upper()}_rouge1_acc": rouge_tmp["rouge1_acc"],
                    #f"{self.name.upper()}_BLEURT_full": bleurt_tmp,
                    f"{self.name.upper()}_BLEU_full": bleu_tmp,
                    #f"{self.name.upper()}_ROUGE_full": rouge_tmp,
                    }

        return metrics

    def eval_saved_file(self, raw_pred_file, prompt_score_file, metric_list=["bleu"]):
        raw_pred_df = pd.read_csv(raw_pred_file).fillna("none")
        prompt_score_df = pd.read_csv(prompt_score_file)
        model_name = raw_pred_file.split("/")[-1].split("_")[0]
        
        metrics = {}
        for prompt in tqdm(prompt_score_df["Prompt"]):
            if "bleu" in metric_list:
                bleu_tmp = lib.utils.bleu_score(raw_pred_df[prompt], self.correct_answer_list, self.incorrect_answer_list)
                metrics.setdefault(f"{model_name}/{self.name.upper()}_BLEU_acc", []).append(bleu_tmp["BLEU_acc"])
                metrics.setdefault(f"{model_name}/{self.name.upper()}_BLEU_full", []).append(bleu_tmp)
            if "rouge" in metric_list:
                rouge_tmp = lib.utils.rouge_score(raw_pred_df[prompt], self.correct_answer_list, self.incorrect_answer_list)
                metrics.setdefault(f"{model_name}/{self.name.upper()}_rouge1_acc", []).append(rouge_tmp["rouge1_acc"])
                metrics.setdefault(f"{model_name}/{self.name.upper()}_ROUGE_full", []).append(rouge_tmp)
            if "bleurt" in metric_list:
                if self.bleurt is None:
                    self.bleurt = load_metric("bleurt")
                bleurt_tmp = lib.utils.bleurt_score(raw_pred_df[prompt], self.correct_answer_list, self.incorrect_answer_list, self.bleurt)
                metrics.setdefault(f"{model_name}/{self.name.upper()}_BLEURT_acc", []).append(bleurt_tmp["BLEURT_acc"])
                metrics.setdefault(f"{model_name}/{self.name.upper()}_BLEURT_full", []).append(bleurt_tmp)
        
        prompt_score_df = pd.read_csv(prompt_score_file)
        for key in metrics:
            prompt_score_df[key] = metrics[key]
        prompt_score_df.to_csv(prompt_score_file, index=False)


class benchmark_socket(benchmark_base):
    def __init__(self, benchmark_name):
        self.name = benchmark_name
        self.task_type_options = {'bragging#brag_achievement': 'For the sentence: "{question_prompt}", is it bragging about an achievement?' + YES_NO_POSTFIX, 
                                  'hahackathon#is_humor': 'For the sentence: "{question_prompt}", is it humorous?' + YES_NO_POSTFIX, 
                                  'tweet_irony': 'For the sentence: "{question_prompt}", is it ironic?' + YES_NO_POSTFIX, 
                                  'sexyn': 'For the sentence: "{question_prompt}", is it sexist?' + YES_NO_POSTFIX,
                                  'tweet_offensive': 'For the sentence: "{question_prompt}", is it offensive?' + YES_NO_POSTFIX,
                                  'complaints': 'For the sentence: "{question_prompt}", is it a complaint?' + YES_NO_POSTFIX,
                                  'empathy#empathy_bin': 'For the sentence: "{question_prompt}", is it expressing empathy?' + YES_NO_POSTFIX,
                                  'stanfordpoliteness': 'For the sentence: "{question_prompt}", is it polite?' + YES_NO_POSTFIX,
                                  'rumor#rumor_bool': 'For the sentence: "{question_prompt}", is it a rumor?' + YES_NO_POSTFIX}
        self.task_type = self.name[len("socket_"):]
        assert self.task_type in self.task_type_options
        data = load_dataset('Blablablab/SOCKET',self.task_type)["sockette"]
        self.data_df = pd.DataFrame({"text": data["text"], "label": data["label"], "task_type": self.name})

        # Some benchmark labels are reversed
        if self.task_type in ["stanfordpoliteness"]:
            self.data_df["label"] = [1 if label_tmp == 0 else 0 for label_tmp in list(self.data_df["label"])]

        self.question_list = self.data_df["text"]
        self.true_label_list = list(self.data_df["label"])
    
    def get_user_prompt(self, args):
        if args.cot == 1:
            return self.task_type_options[self.task_type].replace(YES_NO_POSTFIX, YES_NO_COT_POSTFIX)
        else:
            return self.task_type_options[self.task_type]


    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, args, result_type="yes_no")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            full_true_label_list, full_pred_label_list = [], []
            no_error_true_label_list, no_error_pred_label_list = [], []
            for idx in range(len(pred_label_list)):
                if pred_label_list[idx] not in [0, 1]:
                    full_true_label_list.append(self.true_label_list[idx])
                    full_pred_label_list.append(0)
                else:
                    full_true_label_list.append(self.true_label_list[idx])
                    full_pred_label_list.append(pred_label_list[idx])
                    no_error_true_label_list.append(self.true_label_list[idx])
                    no_error_pred_label_list.append(pred_label_list[idx])

            metrics = {f"{self.name.upper()}_f1": f1_score(full_true_label_list, full_pred_label_list),
                    f"{self.name.upper()}_f1_no_error": f1_score(no_error_true_label_list, no_error_pred_label_list),
                    f"{self.name.upper()}_error": error_num}

        return metrics


def init_benchmark(name="mmlu") -> benchmark_base:
    if name == "mmlu":
        return benchmark_mmlu()
    elif name == "arc":
        return benchmark_arc()
    elif name == "hellaswag":
        return benchmark_hellaswag()
    elif name == "truthfulqa":
        return benchmark_truthfulqa()
    elif "socket" in name:
        return benchmark_socket(name)