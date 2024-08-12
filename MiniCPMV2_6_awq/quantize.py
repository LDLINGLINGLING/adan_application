

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/root/ld/ld_model_pretrain/MiniCPM-V-2_6'
quant_path = '/root/ld/ld_model_pretrain/MiniCPM-V-2_6_awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,device_map={"": "cuda:0"})

# Define data loading methods
def load_alpaca():
    data = load_dataset('/root/ld/pull_request/MiniCPM/quantize/quantize_data/alpaca', split="train")

    # concatenate data
    def concatenate_data(x):
        msgs=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": x['input']},{"role": "system", "content": x['output']}]
        data=tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"text": data}
    
    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]][:1000]

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')