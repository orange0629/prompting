from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm

llama2_template = '''<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]'''
llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mixtral_template = '''<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]'''
dbrx_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''
jamba_template = '''<|startoftext|>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''
gemma_template = '''<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n'''
commandR_template = '''<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'''
llm_template_dict = {"llama-2": llama2_template, "llama-3": llama3_template, "mistral": mixtral_template, "mixtral": mixtral_template, "dbrx": dbrx_template, "jamba": jamba_template, "qwen": qwen_template, "gemma": gemma_template, "command-r": commandR_template}


class inference_model:
    def __init__(self, model_dir, use_vllm=True, cache_dir=None, BATCH_SIZE=1, multi_thread=1):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.use_vllm = use_vllm
        self.BATCH_SIZE = BATCH_SIZE

        self.model_name = model_dir.split("/")[-1] if "/" in model_dir else model_dir
        self.model_type = "llama"
        for key in llm_template_dict:
            if key in self.model_name.lower():
                self.model_type = key

        if use_vllm:
            if self.model_type == "command-r":
                self.model = LLM(model=model_dir, download_dir=cache_dir, tensor_parallel_size=multi_thread)
            elif "nemo" in self.model_dir.lower():
                self.model = LLM(model=model_dir, download_dir=cache_dir, tensor_parallel_size=multi_thread, max_model_len=4096)
            else:
                self.model = LLM(model=model_dir, download_dir=cache_dir, trust_remote_code=True, tensor_parallel_size=multi_thread)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                                        cache_dir=cache_dir,
                                                        padding_side='left',
                                                        )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                            llm_int8_skip_modules=["mamba"])
            
            self.model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                        cache_dir=cache_dir,
                                                        trust_remote_code=True,
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation="flash_attention_2",
                                                        #quantization_config=quantization_config
                                                        )
    
    def generate(self, answer_prompts, max_token_len, use_tqdm=False, return_length=False):
        if self.use_vllm:
            vllm_outputs = self.model.generate(answer_prompts, sampling_params=SamplingParams(max_tokens=max_token_len, temperature=0), use_tqdm=use_tqdm)
            outputs = [output.outputs[0].text for output in vllm_outputs]
            outputs_length = [len(output.outputs[0].token_ids) for output in vllm_outputs]
        else:
            outputs = []
            outputs_length = []
            for idx in tqdm(range(0, len(answer_prompts), self.BATCH_SIZE)):
                ques_batch = answer_prompts[idx:(idx+self.BATCH_SIZE)]
                ques_batch_tokenized = self.tokenizer(ques_batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
                answ_ids = self.model.generate(**ques_batch_tokenized.to('cuda'), max_new_tokens=max_token_len, pad_token_id=self.tokenizer.pad_token_id)
                answ_ids = answ_ids[:, ques_batch_tokenized['input_ids'].shape[1]:] # Only consider new generated stuff
                outputs.extend(self.tokenizer.batch_decode(answ_ids, skip_special_tokens=True))
                outputs_length.extend([len(answ_id) for answ_id in answ_ids])
        if return_length:
            return outputs, outputs_length
        else:
            return outputs

    def get_prompt_template(self):
        return llm_template_dict[self.model_type]