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
            "truthfulqa": "./data/benchmark/truthfulqa/TruthfulQA.csv",
            "hitom": "./data/benchmark/hitom/Hi-ToM_data.json",
            "edos_taska": "./data/benchmark/edos/edos_labelled_aggregated_1000.csv",
            "edos_taskbc": "./data/benchmark/edos/edos_labelled_sexist.csv",
            "ifeval": "./data/benchmark/ifeval/input_data.jsonl",}
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

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        return dict()
    
    def get_user_prompt(self, args):
        if args.cot >= 1:
            return MULTIPLE_CHOICE_COT_USER_PROMPT
        else:
            return MULTIPLE_CHOICE_DEFAULT_USER_PROMPT
    
    def get_max_token_len(self, args):
        if args.cot != 0:
            return 512
        else:
            return 16

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
    
    def get_max_token_len(self, args):
        return 64


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
        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="yes_no")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            metrics = lib.utils.custom_f1_score(self.true_label_list, pred_label_list, self.name.upper())

        return metrics

class benchmark_hitom(benchmark_base):
    def __init__(self):
        self.name = "hitom"
        self.data_df = pd.json_normalize(pd.read_json(data_dir[self.name])['data'])

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"Story:\n{item['story'].strip()}\nQuestion: {item['question']}\nChoices: {item['choices']}"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["answer"])


    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, args, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            assert len(self.true_label_list) == len(pred_label_list)
            acc_num = 0
            for idx in range(len(self.true_label_list)):
                if self.true_label_list[idx].lower() in pred_label_list[idx].lower().replace(" ", "") or self.true_label_list[idx].lower().replace("_", " ") in pred_label_list[idx].lower():
                    acc_num += 1
            metrics = {f"{self.name.upper()}_acc_no_error": acc_num/len(self.true_label_list)}

        return metrics

    def get_user_prompt(self, args):
        return "Read the following story and answer the multiple-choice question. Please provide answer without explanations.\n{question_prompt}\n\nNote: You should assume the following. (1) An agent witnesses everything and every movements before exiting a location. (2) An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. (3) Note that every agent tend to lie. What a character tells others doesn't affect his actual belief. An agent tend to trust a agent that exited the room later than himself. The exit order is known to all agents. (4) Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims."


class benchmark_edos(benchmark_base):
    def __init__(self, benchmark_name):
        self.name = benchmark_name
        self.task_type_options = {'taska': {'prompt': 'For the post: "{question_prompt}", is it sexist?' + YES_NO_POSTFIX, 'col_name': 'label_sexist'}, 
                                  'taskb': {'prompt': 'For the sexist post: "{question_prompt}", classify it into one of the following 4 sexism categories:\n(1) threats, plans to harm and incitement\n(2) derogation\n(3) animosity\n(4) prejudiced discussions. Reply with only the name of category.', 'col_name': 'label_category'}, 
                                  'taskc': {'prompt': 'For the sentence: "{question_prompt}", is it ironic?' + YES_NO_POSTFIX, 'col_name': 'label_vector'}}
        self.task_type = self.name[len("edos_"):]
        assert self.task_type in self.task_type_options
        if "taska" in self.name:
            self.data_df = pd.read_csv(data_dir["edos_taska"])
            self.true_label_list = [0 if tmp == "not sexist" else 1 for tmp in list(self.data_df[self.task_type_options[self.task_type]['col_name']])]
        else:
            self.data_df = pd.read_csv(data_dir["edos_taskbc"])
            self.true_label_list = self.data_df[self.task_type_options[self.task_type]['col_name']]
        self.question_list = self.data_df["text"]
        
    
    def get_user_prompt(self, args):
        if args.cot == 1:
            return self.task_type_options[self.task_type]['prompt'].replace(YES_NO_POSTFIX, YES_NO_COT_POSTFIX)
        else:
            return self.task_type_options[self.task_type]['prompt']


    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        if self.task_type == "taska":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="yes_no")
        elif self.task_type == "taskb":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="raw")
        elif self.task_type == "taskc":
            pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if self.task_type == "taska":
                metrics = lib.utils.custom_f1_score(self.true_label_list, pred_label_list, self.name.upper())
            elif self.task_type == "taskb":
                classify_options = {"threats": "1. threats, plans to harm and incitement", "derogation": "2. derogation", "animosity": "3. animosity", "prejudiced discussions": "4. prejudiced discussions"}
                for idx in range(len(pred_label_list)):
                    for sub_option in classify_options:
                        if sub_option in pred_label_list[idx].lower():
                            pred_label_list[idx] = classify_options[sub_option]
                metrics = {f"{self.name.upper()}_f1_no_error": f1_score(self.true_label_list, pred_label_list, average="macro")}
            elif self.task_type == "taskc":
                pass

        return metrics

class benchmark_ifeval(benchmark_base):
    def __init__(self):
        self.name = "ifeval"
        self.data_df = pd.read_json(data_dir[self.name], lines=True)
        #self.data_df = self.data_df[self.data_df["instruction_id_list"].apply(lambda x: "language:response_language" not in x)]

        self.question_list = self.data_df["prompt"]
        self.true_label_list = []
    
    def get_user_prompt(self, args):
        return QA_DEFAULT_USER_PROMPT

    def eval_question_list(self, pred_text_list, args, save_intermediate=("all", "", "")):
        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, args, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2])

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            assert len(self.data_df) == len(pred_label_list)
            result_data_dict = dict(zip(list(self.data_df["prompt"]), pred_label_list))
            import lib.ifeval.evaluation_main
            metrics = {f"{self.name.upper()}_acc_no_error": lib.ifeval.evaluation_main.run_eval(data_dir[self.name], result_data_dict)["acc"]}
        
        return metrics
    
    def get_max_token_len(self, args):
        return 512


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
    elif name == "hitom":
        return benchmark_hitom()
    elif "edos" in name:
        return benchmark_edos(name)
    elif "ifeval" in name:
        return benchmark_ifeval()