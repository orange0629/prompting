from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

class hfmodel:
    def __init__(self, model_dir, cache_dir, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE

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
    
    def generate(self, answer_prompts):
        outputs = []
        for idx in tqdm(range(0, len(answer_prompts), self.BATCH_SIZE)):
            ques_batch = answer_prompts[idx:(idx+self.BATCH_SIZE)]
            ques_batch_tokenized = self.tokenizer(ques_batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
            answ_ids = self.model.generate(**ques_batch_tokenized.to('cuda'), max_new_tokens=30, pad_token_id=self.tokenizer.pad_token_id)
            outputs.extend(self.tokenizer.batch_decode(answ_ids, skip_special_tokens=True))
        return outputs