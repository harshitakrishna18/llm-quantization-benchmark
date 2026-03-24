import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_baseline(model_name: str):
    start = time.time()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    load_time = time.time()-start
    return tokenizer, model, load_time

def load_8bit(model_name: str):
    start = time.time()
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config = bnb_config, 
        device_map="auto")

    load_time = time.time()-start

    return tokenizer, model, load_time

def load_4bit(model_name: str):
    start = time.time()
    # Store weights in 4-bit
    # Use NF4 (smarter 4-bit format for LLM weights since LLM weights are not uniformly distributed)
    # Do actual math in fp16
    # Compress the scale metadata too
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16, 
        bnb_4bit_quant_type = 'nf4', 
        bnb_4bit_use_double_quant= True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config = bnb_config,
        device_map = 'auto')

    
    load_time = time.time()-start

    return tokenizer, model, load_time


