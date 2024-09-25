import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import load_metric, load_dataset
from tqdm import tqdm
import lib.utils
import os
import re
import json

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_dir = {"mmlu": "./data/benchmark/mmlu/mmlu_mingqian.csv", 
            "arc": "./data/benchmark/arc/ARC-Challenge-Test.csv",
            "hellaswag": "./data/benchmark/hellaswag/hellaswag_train.jsonl",
            "truthfulqa": "./data/benchmark/truthfulqa/TruthfulQA.csv",
            "hitom": "./data/benchmark/hitom/Hi-ToM_data.json",
            "edos_taska": "./data/benchmark/edos/edos_labelled_aggregated_1000.csv",
            "edos_taskbc": "./data/benchmark/edos/edos_labelled_sexist.csv",
            "ifeval": "./data/benchmark/ifeval/input_data.jsonl",
            "bbh": "./data/benchmark/bbh/",
            "brainteaser": "./data/benchmark/brainteaser/brainteaser_semantic-reconstruction.csv",
            "gsm8k": "./data/benchmark/gsm8k/gsm8k_test.csv",}
save_intermediate_dir = os.path.join(project_root_dir, "./results/benchmark")

#MULTIPLE_CHOICE_DEFAULT_USER_PROMPT = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"
MULTIPLE_CHOICE_DEFAULT_USER_PROMPT = 'The following is a multiple choice question (with answers).\n{question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).'
MULTIPLE_CHOICE_COT_USER_PROMPT = "The following is a multiple choice question (with answers). Think carefully step by step. Describe your reasoning in steps and then output the option letter at the very end.\n{question_prompt}"

# YES_NO_POSTFIX = " Reply with only yes or no."
#YES_NO_POSTFIX = " Show your final answer (Yes or No only) bracketed between <answer> and </answer>."
YES_NO_POSTFIX = '\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).'
YES_NO_COT_POSTFIX = " Think carefully step by step. Describe your reasoning in steps and then output yes or no at the very end."

#QA_DEFAULT_USER_PROMPT = "{question_prompt} Show your final answer bracketed between <answer> and </answer>."
QA_DEFAULT_USER_PROMPT = '{question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question.'

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

class benchmark_base:
    def __init__(self, cot):
        self.name = "base"
        self.data_df, self.question_list, self.true_label_list = pd.DataFrame(), [], []
        self.cot = cot
    
    def save_intermediate(self, pred_label_list, model_name, column_name, eval_range=None):
        if not os.path.exists(save_intermediate_dir):
            os.makedirs(save_intermediate_dir)
        save_dir_tmp = f"{save_intermediate_dir}/{model_name}_{self.name}_results.csv"
        try:
            save_df = pd.read_csv(save_dir_tmp)
        except:
            save_df = self.data_df.copy()
        if eval_range is None:
            save_df[column_name] = pred_label_list
        else:
            save_df.loc[eval_range, column_name] = pred_label_list
        save_df.to_csv(save_dir_tmp, index=False)
    
    def clean_text(self, text):
        pattern = r"[^a-zA-Z0-9 !#$%&()*'\"+,.:;<=>?@_{|}-]"
        cleaned_text = re.sub(pattern, ' ', text)
        return re.sub("\s\s+" , " ", cleaned_text).strip()

    def result_list_preprocessing(self, pred_text_list, result_type="multiple_choice"):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            text = self.clean_text(pred_text)

            # Answer tag extraction
            start = text.find("<answer>") + len("<answer>") if text.find("<answer>") != -1 else 0
            end = text.find("</answer>") if text.find("</answer>") != -1 else len(text)
            text = text[start:end]
            start = text.rfind("Answer:") + len("Answer:") if text.rfind("Answer:") != -1 else -5 #Only tolerate 5 chars
            text = text[start:]
            
            if result_type == "multiple_choice":
                pattern = re.compile(r'[ABCD]')
                matches = list(pattern.finditer(text))
                if matches:
                    if self.cot != 0:
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
                    if self.cot != 0:
                        pred_label_list.append(int(matches[-1].group().lower() == "yes"))
                    else:
                        pred_label_list.append(int(matches[0].group().lower() == "yes"))
                else:
                    pred_label_list.append(text)
                    error_num += 1
            else:
                pred_label_list.append(text)

        return pred_label_list, error_num
    
    def load_question_list(self):
        return self.question_list
    
    def load_random_question_list(self, num_q=None, split="all"):
        train_indices, test_indices = train_test_split(list(range(len(self.question_list))), test_size=0.4, random_state=42)
        if split == "all":
            if num_q is None:
                return self.question_list, None
            else:
                rand_idx = random.sample(range(len(self.question_list)), num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx
        elif split == "train":
            if num_q is None:
                return [self.question_list[i] for i in train_indices], train_indices
            else:
                rand_idx = random.sample(train_indices, num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx
        elif split == "test":
            if num_q is None:
                return [self.question_list[i] for i in test_indices], test_indices
            else:
                rand_idx = random.sample(test_indices, num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        return dict()
    
    def get_user_prompt(self):
        if self.cot >= 1:
            return MULTIPLE_CHOICE_COT_USER_PROMPT
        else:
            return MULTIPLE_CHOICE_DEFAULT_USER_PROMPT
    
    def get_max_token_len(self):
        if self.cot != 0:
            return 512
        else:
            return 16

class benchmark_mmlu(benchmark_base):
    def __init__(self, cot):
        self.name = "mmlu"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['question'].strip()}\nA. {item['option1']}\nB. {item['option2']}\nC. {item['option3']}\nD. {item['option4']}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["true_option"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = num2letter[self.true_label_list[idx]]

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),
                    f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    f"{self.name.upper()}_error": error_num}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_arc(benchmark_base):
    def __init__(self, cot):
        self.name = "arc"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = item["question"].strip().replace("(1)", "(A)").replace("(2)", "(B)").replace("(3)", "(C)").replace("(4)", "(D)") + "\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["AnswerKey"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = self.true_label_list[idx].upper().strip()
            if self.true_label_list[idx] in num2letter:
                self.true_label_list[idx] = num2letter[self.true_label_list[idx]]
    

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),
                    f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    f"{self.name.upper()}_error": error_num}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_hellaswag(benchmark_base):
    def __init__(self, cot):
        self.name = "hellaswag"
        self.data_df = pd.read_json(path_or_buf=os.path.join(project_root_dir, data_dir[self.name]), lines=True).sample(n=1000, random_state=42).reset_index(drop=True)
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['ctx'].strip()}\nA. {item['endings'][0]}\nB. {item['endings'][1]}\nC. {item['endings'][2]}\nD. {item['endings'][3]}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["label"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = num2letter[int(self.true_label_list[idx])+1]


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),
                    f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    f"{self.name.upper()}_error": error_num}
            
            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_truthfulqa(benchmark_base):
    def __init__(self, cot):
        self.name = "truthfulqa"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = self.data_df["Question"]
        self.true_label_list = list(self.data_df["Best Answer"])

        self.correct_answer_list = [lib.utils.split_multi_answer(text, add_no_comment=True) for text in self.data_df["Correct Answers"]]
        self.incorrect_answer_list = [lib.utils.split_multi_answer(text) for text in self.data_df["Incorrect Answers"]]

        self.bleurt = None
    
    def get_user_prompt(self):
        return QA_DEFAULT_USER_PROMPT

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                bleu_tmp = lib.utils.bleu_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list, return_error_idx)
                #rouge_tmp = lib.utils.rouge_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list)
                #if self.bleurt is None:
                #    self.bleurt = load_metric("bleurt")
                #bleurt_tmp = lib.utils.bleurt_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list, self.bleurt)
            else:
                bleu_tmp = lib.utils.bleu_score(pred_label_list, [self.correct_answer_list[i] for i in eval_range], [self.incorrect_answer_list[i] for i in eval_range], return_error_idx)
            

            metrics = {#f"{self.name.upper()}_BLEURT_acc": bleurt_tmp["BLEURT_acc"],
                    f"{self.name.upper()}_BLEU_acc": bleu_tmp["BLEU_acc"],
                    #f"{self.name.upper()}_rouge1_acc": rouge_tmp["rouge1_acc"],
                    #f"{self.name.upper()}_BLEURT_full": bleurt_tmp,
                    f"{self.name.upper()}_BLEU_full": bleu_tmp,
                    #f"{self.name.upper()}_ROUGE_full": rouge_tmp,
                    }

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = bleu_tmp["BLEU_error_idx"]

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
    
    def get_max_token_len(self):
        return 64


class benchmark_socket(benchmark_base):
    def __init__(self, benchmark_name, cot):
        self.name = benchmark_name
        self.cot = cot
        self.task_type_options = {'bragging#brag_achievement': 'For the sentence: "{question_prompt}", is it bragging about an achievement?' + YES_NO_POSTFIX, 
                                  'hahackathon#is_humor': 'For the sentence: "{question_prompt}", is it humorous?' + YES_NO_POSTFIX, 
                                  'tweet_irony': 'For the sentence: "{question_prompt}", is it ironic?' + YES_NO_POSTFIX, 
                                  'sexyn': 'For the sentence: "{question_prompt}", is it sexist?' + YES_NO_POSTFIX,
                                  'tweet_offensive': 'For the sentence: "{question_prompt}", is it offensive?' + YES_NO_POSTFIX,
                                  'complaints': 'For the sentence: "{question_prompt}", is it a complaint?' + YES_NO_POSTFIX,
                                  'empathy#empathy_bin': 'For the sentence: "{question_prompt}", is it expressing empathy?' + YES_NO_POSTFIX,
                                  'stanfordpoliteness': 'For the sentence: "{question_prompt}", is it polite?' + YES_NO_POSTFIX,
                                  'rumor#rumor_bool': 'For the sentence: "{question_prompt}", is it a rumor?' + YES_NO_POSTFIX,
                                  'empathy#distress_bin': 'For the sentence: "{question_prompt}", is it showing distress?' + YES_NO_POSTFIX, # Newly added
                                  "jigsaw#insult":  'For the sentence: "{question_prompt}", is it an insult?' + YES_NO_POSTFIX,
                                  }
        self.task_type = self.name[len("socket_"):]
        assert self.task_type in self.task_type_options
        data = load_dataset('Blablablab/SOCKET',self.task_type, trust_remote_code=True)["sockette"]
        self.data_df = pd.DataFrame({"text": data["text"], "label": data["label"], "task_type": self.name})

        # Some benchmark labels are reversed
        if self.task_type in ["stanfordpoliteness"]:
            self.data_df["label"] = [1 if label_tmp == 0 else 0 for label_tmp in list(self.data_df["label"])]

        self.question_list = self.data_df["text"]
        self.true_label_list = list(self.data_df["label"])
    
    def get_user_prompt(self):
        if self.cot == 1:
            return self.task_type_options[self.task_type].replace(YES_NO_POSTFIX, YES_NO_COT_POSTFIX)
        else:
            return self.task_type_options[self.task_type]


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="yes_no")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = lib.utils.custom_f1_score(local_true_label_list, pred_label_list, self.name.upper())

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_hitom(benchmark_base):
    def __init__(self, cot):
        self.name = "hitom"
        self.data_df = pd.json_normalize(pd.read_json(os.path.join(project_root_dir, data_dir[self.name]))['data'])
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"Story:\n{item['story'].strip()}\nQuestion: {item['question']}\nChoices: {item['choices']}"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["answer"])


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            assert len(local_true_label_list) == len(pred_label_list)
            error_idx = []
            for idx in range(len(local_true_label_list)):
                if not (local_true_label_list[idx].lower() in pred_label_list[idx].lower().replace(" ", "") or local_true_label_list[idx].lower().replace("_", " ") in pred_label_list[idx].lower()):
                    error_idx.append(idx)
            metrics = {f"{self.name.upper()}_acc_no_error": 1 - len(error_idx)/len(local_true_label_list)}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = error_idx

        return metrics

    def get_user_prompt(self):
        return '''Read the following story and answer the multiple-choice question. \n{question_prompt}\n\nNote: You should assume the following. (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. An agent tend to trust a agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims.\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option content only).'''


class benchmark_edos(benchmark_base):
    def __init__(self, benchmark_name, cot):
        self.name = benchmark_name
        self.cot = cot
        self.task_type_options = {'taska': {'prompt': 'For the post: "{question_prompt}", is it sexist?' + YES_NO_POSTFIX, 'col_name': 'label_sexist'}, 
                                  'taskb': {'prompt': 'For the sexist post: "{question_prompt}", classify it into one of the following 4 sexism categories:\n(1) threats, plans to harm and incitement\n(2) derogation\n(3) animosity\n(4) prejudiced discussions. Reply with only the name of category.', 'col_name': 'label_category'}, 
                                  'taskc': {'prompt': 'For the sentence: "{question_prompt}", is it ironic?' + YES_NO_POSTFIX, 'col_name': 'label_vector'}}
        self.task_type = self.name[len("edos_"):]
        assert self.task_type in self.task_type_options
        if "taska" in self.name:
            self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir["edos_taska"]))
            self.true_label_list = [0 if tmp == "not sexist" else 1 for tmp in list(self.data_df[self.task_type_options[self.task_type]['col_name']])]
        else:
            self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir["edos_taskbc"]))
            self.true_label_list = self.data_df[self.task_type_options[self.task_type]['col_name']]
        self.question_list = self.data_df["text"]
        
    
    def get_user_prompt(self):
        if self.cot == 1:
            return self.task_type_options[self.task_type]['prompt'].replace(YES_NO_POSTFIX, YES_NO_COT_POSTFIX)
        else:
            return self.task_type_options[self.task_type]['prompt']


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        if self.task_type == "taska":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="yes_no")
        elif self.task_type == "taskb":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        elif self.task_type == "taskc":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            if self.task_type == "taska":
                metrics = lib.utils.custom_f1_score(local_true_label_list, pred_label_list, self.name.upper())
                if return_error_idx:
                    metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]
            elif self.task_type == "taskb":
                classify_options = {"threats": "1. threats, plans to harm and incitement", "derogation": "2. derogation", "animosity": "3. animosity", "prejudiced discussions": "4. prejudiced discussions"}
                for idx in range(len(pred_label_list)):
                    for sub_option in classify_options:
                        if sub_option in pred_label_list[idx].lower():
                            pred_label_list[idx] = classify_options[sub_option]
                metrics = {f"{self.name.upper()}_f1_no_error": f1_score(local_true_label_list, pred_label_list, average="macro", zero_division=0.0)}
            elif self.task_type == "taskc":
                pass

        return metrics

class benchmark_ifeval(benchmark_base):
    def __init__(self, cot):
        self.name = "ifeval"
        self.data_df = pd.read_json(os.path.join(project_root_dir, data_dir[self.name]), lines=True)
        #self.data_df = self.data_df[self.data_df["instruction_id_list"].apply(lambda x: "language:response_language" not in x)]
        self.cot = cot

        self.question_list = self.data_df["prompt"]
        self.true_label_list = []
    
    def get_user_prompt(self):
        return QA_DEFAULT_USER_PROMPT

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                result_data_dict = dict(zip(list(self.data_df["prompt"]), pred_label_list))
            else:
                result_data_dict = dict(zip([self.data_df["prompt"][i] for i in eval_range], pred_label_list))
            import lib.ifeval.evaluation_main
            metrics = {f"{self.name.upper()}_acc_no_error": lib.ifeval.evaluation_main.run_eval(os.path.join(project_root_dir, data_dir[self.name]), result_data_dict, eval_range)["acc"]}
        
        return metrics
    
    def get_max_token_len(self):
        return 512


class benchmark_bbh(benchmark_base):
    def __init__(self, benchmark_name, cot):
        self.name = benchmark_name
        self.cot = cot
        # self.task_type_options = {'boolean_expressions': 'Evaluate the result of the following Boolean expression. Show your final answer (True or False only) bracketed between <answer> and </answer>.\nQ: {question_prompt}', 
        #                           'causal_judgement': 'Answer the following question about causal attribution. How would a typical person answer each of the following questions about causation? Show your final answer (Yes or No only) bracketed between <answer> and </answer>.\nQ: {question_prompt}', 
        #                           'date_understanding': 'Infer the date from context. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}', 
        #                           'disambiguation_qa': 'Clarify the meaning of sentences with ambiguous pronouns. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'dyck_languages': 'Correctly close a Dyck-n word. Show your final answer bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'formal_fallacies': 'Distinguish deductively valid arguments from formal fallacies. Be cautious, some arguments may have premises that are nonsensical or contradictory. In such cases, simply focus on determining whether or not the conclusion is supported by the premises, regardless of their content.\nRead each argument and provided premises carefully and attentively. If the argument can be demonstrated to be invalid based on the premises, respond with "invalid," otherwise, answer "valid." Show your final answer (valid or invalid only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'geometric_shapes': 'Name geometric shapes from their SVG paths. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'hyperbaton': 'Order adjectives correctly in English sentences. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'logical_deduction_five_objects': 'A logical deduction task which requires deducing the order of a sequence of objects. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'logical_deduction_seven_objects': 'A logical deduction task which requires deducing the order of a sequence of objects. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'logical_deduction_three_objects': 'A logical deduction task which requires deducing the order of a sequence of objects. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'movie_recommendation': "Recommend movies similar to the given list of movies. Let's think step by step. First, let's identify the common themes or genres of the given movies. Then, let's look at the options and choose the one that best fits the common themes or genres. If none of the options fit the common themes or genres perfectly, let's choose the option that is most similar to the given movies in terms of its popularity and well-knownness. Finally, let's bracket the final answer option between <answer> and </answer>.Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}",
        #                           'multistep_arithmetic_two': 'Solve the following math problems by following the steps in the order of operations:\n\n1. When multiplying or dividing two negative numbers, the result will be positive.\n2. When multiplying or dividing a positive and a negative number, the result will be negative.\n3. When adding or subtracting a negative number, it is the same as adding or subtracting its positive counterpart. Show your final answer (a number only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'navigate': 'Given a set of instructions, determine whether following those instructions will take you back to the exact same spot you started from. Keep in mind any movements, including turns and the direction of any steps. For example, if you take 2 steps forward and then 2 steps backward, you will end up in the same spot. Determine your answer by saying "yes" or "no". Show your final answer (Yes or No only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'object_counting': 'Questions that involve enumerating objects and asking the model to count them. Show your final answer (a number only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'penguins_in_a_table': 'Answer questions about a table of penguins and their attributes. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'reasoning_about_colored_objects': 'Answer extremely simple questions about the colors of objects on a surface. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'ruin_names': "Select the humorous edit that 'ruins' the input movie or musical artist name. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}",
        #                           #'salient_translation_error_detection': 'For the sentence: "{question_prompt}", is it a rumor?',
        #                           'snarks': 'Determine which of two sentences is sarcastic. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'sports_understanding': 'Determine whether an artificially constructed sentence relating to sports is plausible or not. Show your final answer (yes or no only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'temporal_sequences': 'Answer questions about which times certain events could have occurred. To solve this problem, we can break it down into smaller steps. The first step is to find the time when the person woke up. Once we have that information, we can then proceed to the next step, which is to identify the earliest time slot that has not been accounted for. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'tracking_shuffled_objects_five_objects': "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. Let's approach this task in a systematic manner. First, let's identify all the different objects that are being swapped in the context. Then, let's trace the swaps and keep track of the objects as they change hands. Finally, let's use our understanding of the swaps and the initial positions of the objects to answer the question. We also need to make sure that all the information necessary to answer the question is contained in the context and check that the number of players is the same as the number of objects. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}",
        #                           'tracking_shuffled_objects_seven_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'tracking_shuffled_objects_three_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. We will methodically address this by breaking it down into manageable portions. We will monitor the final status of every entity (e.g., Alice, Bob, and Claire) after the transactions noted in the problem. We will also monitor the flow of the transactions. Show your final answer option bracketed between <answer> and </answer> at the end.\nQ: {question_prompt}',
        #                           'web_of_lies': 'Evaluate a random boolean function expressed as a word problem. Show your final answer (Yes or No only) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           'word_sorting': 'Sort a list of words. Show your final answer (only words seperated by whitespace) bracketed between <answer> and </answer>.\nQ: {question_prompt}',
        #                           }
        self.task_type_options = {'boolean_expressions': 'Evaluate the result of the following Boolean expression.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (True or False only).', 
                                  'causal_judgement': 'Answer the following question about causal attribution.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).', 
                                  'date_understanding': 'Infer the date from context.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).', 
                                  'disambiguation_qa': 'Clarify the meaning of sentences with ambiguous pronouns.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'dyck_languages': 'Correctly close a Dyck-n word.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question',
                                  'formal_fallacies': 'Distinguish deductively valid arguments from formal fallacies.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (valid or invalid only).',
                                  'geometric_shapes': 'Name geometric shapes from their SVG paths.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'hyperbaton': 'Order adjectives correctly in English sentences.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'logical_deduction_five_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'logical_deduction_seven_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'logical_deduction_three_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'movie_recommendation': '''Recommend movies similar to the given list of movies.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
                                  'multistep_arithmetic_two': 'Solve multi-step arithmetic problems.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (a number only).',
                                  'navigate': 'Given a series of navigation instructions, determine whether one would end up back at the starting point.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).',
                                  'object_counting': 'Questions that involve enumerating objects and asking the model to count them.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (a number only).',
                                  'penguins_in_a_table': 'Answer questions about a table of penguins and their attributes.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question',
                                  'reasoning_about_colored_objects': 'Answer extremely simple questions about the colors of objects on a surface.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'ruin_names': '''Select the humorous edit that 'ruins' the input movie or musical artist name.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
                                  #'salient_translation_error_detection': 'For the sentence: "{question_prompt}", is it a rumor?',
                                  'snarks': 'Determine which of two sentences is sarcastic.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'sports_understanding': 'Determine whether an artificially constructed sentence relating to sports is plausible or not.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (yes or no only).',
                                  'temporal_sequences': 'Task description: Answer questions about which times certain events could have occurred.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'tracking_shuffled_objects_five_objects': '''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
                                  'tracking_shuffled_objects_seven_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'tracking_shuffled_objects_three_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
                                  'web_of_lies': 'Evaluate a random boolean function expressed as a word problem.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).',
                                  'word_sorting': 'Sort a list of words.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (only words seperated by whitespace).',
                                  }
        # self.task_type_options = {'boolean_expressions': 'Evaluate the result of the following Boolean expression.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (True or False only).', 
        #                           'causal_judgement': 'Answer the following question about causal attribution. How would a typical person answer each of the following questions about causation?\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).', 
        #                           'date_understanding': 'Infer the date from context.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).', 
        #                           'disambiguation_qa': 'Clarify the meaning of sentences with ambiguous pronouns.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'dyck_languages': 'Correctly close a Dyck-n word.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question',
        #                           'formal_fallacies': 'Distinguish deductively valid arguments from formal fallacies. Be cautious, some arguments may have premises that are nonsensical or contradictory. In such cases, simply focus on determining whether or not the conclusion is supported by the premises, regardless of their content.\nRead each argument and provided premises carefully and attentively. If the argument can be demonstrated to be invalid based on the premises, respond with "invalid," otherwise, answer "valid."\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (valid or invalid only).',
        #                           'geometric_shapes': 'Name geometric shapes from their SVG paths.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'hyperbaton': 'Order adjectives correctly in English sentences.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'logical_deduction_five_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'logical_deduction_seven_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'logical_deduction_three_objects': 'A logical deduction task which requires deducing the order of a sequence of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'movie_recommendation': '''Recommend movies similar to the given list of movies. Let's think step by step. First, let's identify the common themes or genres of the given movies. Then, let's look at the options and choose the one that best fits the common themes or genres. If none of the options fit the common themes or genres perfectly, let's choose the option that is most similar to the given movies in terms of its popularity and well-knownness. Finally, let's print the final answer option after "Answer:".\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
        #                           'multistep_arithmetic_two': 'Solve the following math problems by following the steps in the order of operations:\n\n1. When multiplying or dividing two negative numbers, the result will be positive.\n2. When multiplying or dividing a positive and a negative number, the result will be negative.\n3. When adding or subtracting a negative number, it is the same as adding or subtracting its positive counterpart.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (a number only).',
        #                           'navigate': 'Given a set of instructions, determine whether following those instructions will take you back to the exact same spot you started from. Keep in mind any movements, including turns and the direction of any steps. For example, if you take 2 steps forward and then 2 steps backward, you will end up in the same spot. Determine your answer by saying "yes" or "no".\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).',
        #                           'object_counting': 'Questions that involve enumerating objects and asking the model to count them.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (a number only).',
        #                           'penguins_in_a_table': 'Answer questions about a table of penguins and their attributes.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question',
        #                           'reasoning_about_colored_objects': 'Answer extremely simple questions about the colors of objects on a surface.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'ruin_names': '''Select the humorous edit that 'ruins' the input movie or musical artist name.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
        #                           #'salient_translation_error_detection': 'For the sentence: "{question_prompt}", is it a rumor?',
        #                           'snarks': 'Determine which of two sentences is sarcastic.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'sports_understanding': 'Determine whether an artificially constructed sentence relating to sports is plausible or not.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (yes or no only).',
        #                           'temporal_sequences': 'Answer questions about which times certain events could have occurred. To solve this problem, we can break it down into smaller steps. The first step is to find the time when the person woke up. Once we have that information, we can then proceed to the next step, which is to identify the earliest time slot that has not been accounted for.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'tracking_shuffled_objects_five_objects': '''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. Let's approach this task in a systematic manner. First, let's identify all the different objects that are being swapped in the context. Then, let's trace the swaps and keep track of the objects as they change hands. Finally, let's use our understanding of the swaps and the initial positions of the objects to answer the question. We also need to make sure that all the information necessary to answer the question is contained in the context and check that the number of players is the same as the number of objects.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).''',
        #                           'tracking_shuffled_objects_seven_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. \nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'tracking_shuffled_objects_three_objects': 'A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps. We will methodically address this by breaking it down into manageable portions. We will monitor the final status of every entity (e.g., Alice, Bob, and Claire) after the transactions noted in the problem. We will also monitor the flow of the transactions.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).',
        #                           'web_of_lies': 'Evaluate a random boolean function expressed as a word problem.\nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).',
        #                           'word_sorting': 'Sort a list of words. \nQ: {question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (only words seperated by whitespace).',
        #                           }
        self.task_type = self.name[len("bbh_"):]
        assert self.task_type in self.task_type_options
        with open(os.path.join(project_root_dir, data_dir["bbh"]) + f"{self.task_type}.json", 'r') as file:
            data = json.load(file)["examples"]
        self.data_df = pd.DataFrame(data)

        self.question_list = self.data_df["input"]
        self.true_label_list = list(self.data_df["target"])


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            assert len(local_true_label_list) == len(pred_label_list)
            correct_idx = []
            for idx in range(len(local_true_label_list)):
                # If it's multiple choice
                pattern_A = r'^\([A-Z]\)$'
                if re.match(pattern_A, local_true_label_list[idx]):
                    letter_A = local_true_label_list[idx][1]
                    pattern_B = r'\b[A-Z]\b'
                    match_B = re.search(pattern_B, pred_label_list[idx], re.MULTILINE)
                    if match_B:
                        if letter_A == match_B.group(0):
                            correct_idx.append(idx)
                else:
                    # If true answer start with special character
                    if bool(re.match(r'^\W', local_true_label_list[idx])):
                        pattern = r'(?<!\w)' + re.escape(local_true_label_list[idx]) + r'(?!\S)'
                    else:
                        pattern = r'\b' + re.escape(local_true_label_list[idx]) + r'\b'
                    if re.search(pattern, pred_label_list[idx], re.IGNORECASE | re.MULTILINE):
                        correct_idx.append(idx)
            metrics = {f"{self.name.upper()}_acc_no_error": len(correct_idx)/len(local_true_label_list)}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [idx for idx in range(len(local_true_label_list)) if idx not in correct_idx]
        return metrics
    
    def get_max_token_len(self):
        return 512


class benchmark_brainteaser(benchmark_base):
    def __init__(self, cot):
        self.name = "brainteaser"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['question'].strip()}\nA. {item['option1']}\nB. {item['option2']}\nC. {item['option3']}\nD. {item['option4']}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["answer"])

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),}
                    #f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    #f"{self.name.upper()}_error": error_num}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics


class benchmark_gsm8k(benchmark_base):
    def __init__(self, cot):
        self.name = "gsm8k"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = list(self.data_df["question"])
        
        self.true_label_list = list(self.data_df["answer"].apply(lambda x: str(x)))

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            assert len(local_true_label_list) == len(pred_label_list)
            correct_idx = []
            for idx in range(len(local_true_label_list)):
                # If true answer start with special character
                if bool(re.match(r'^\W', local_true_label_list[idx])):
                    pattern = r'(?<!\w)' + re.escape(local_true_label_list[idx]) + r'(?!\S)'
                else:
                    pattern = r'\b' + re.escape(local_true_label_list[idx]) + r'\b'
                if re.search(pattern, pred_label_list[idx], re.IGNORECASE | re.MULTILINE):
                    correct_idx.append(idx)
            metrics = {f"{self.name.upper()}_acc_no_error": len(correct_idx)/len(local_true_label_list)}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [idx for idx in range(len(local_true_label_list)) if idx not in correct_idx]
        return metrics


def init_benchmark(name="mmlu", cot=0) -> benchmark_base:
    if name == "mmlu":
        return benchmark_mmlu(cot=cot)
    elif name == "arc":
        return benchmark_arc(cot=cot)
    elif name == "hellaswag":
        return benchmark_hellaswag(cot=cot)
    elif name == "truthfulqa":
        return benchmark_truthfulqa(cot=cot)
    elif "socket" in name:
        return benchmark_socket(name, cot=cot)
    elif name == "hitom":
        return benchmark_hitom(cot=cot)
    elif "edos" in name:
        return benchmark_edos(name, cot=cot)
    elif "ifeval" in name:
        return benchmark_ifeval(cot=cot)
    elif "bbh" in name:
        return benchmark_bbh(name, cot=cot)
    elif "brainteaser" in name:
        return benchmark_brainteaser(cot=cot)
    elif "gsm8k" in name:
        return benchmark_gsm8k(cot=cot)