input_prompt="""现在我给你一个问题和一个答案，请你帮我对这个问题和答案进行思考后改写，改写的要求如下:{},我会给你一个示例：{},请你根据示例和要求对以下输入的问题和答案进行思考后修改:「输入问题」{}「输入问题」\n「输入答案」{}「输入答案」"""
requirement=""""1. 如果给的问题并不是标准的问题，而是一个描述，请你基于这个问题和答案重写，保持回复的意思不变。
                2. 如果问题过于笼统，请你基于这个问题进行修改成一个比较真实的问题，但请保持问题和回复的答案意思不变。
                3. 输出的问题请和答案请尽量口语话，不要过于书面语
                4. 请将问题变为对第二人称的提问，比如‘他吃饭了么’？需要改成‘你吃饭了么’
                5. 特别重要的是有一部分答案是对问题的续写，请你将其修改成问答形式或者对话形式，要求遵照以上
                6. 进行「思考」后修改，修改后的问题前后用「修改后的问题」包裹，修改后的答案用「修改后的答案」包裹"""
example = """
「输入问题」他进来时，把所有人都吵醒了。「输入问题」
「输入答案」他进来时，大声喧哗，把所有人都吵醒了。他充满活力，喜欢在人群中表现自己，不怕引起注意。「输入答案结束」
「思考流程」输入的答案“他进来时，大声喧哗，把所有人都吵醒了。他充满活力，喜欢在人群中表现自己，不怕引起注意。”是在输入问题上的续写，原始输入的问题也是无效的。我需要对其进行改写成第一人称，而且需要是对话形式的问题。「思考」
「修改后的问题」你进到寝室的时候把吵醒了睡觉的人么？「修改后的问题」
「修改后的答案」 我进到寝室里的时候，想和大家分享我比赛成功的喜悦，充满活力的跟大家讲着比赛的策略和经过，但是太大声了，把所有人都吵醒了。「修改后的答案」"""

model_path = "/root/ld/ld_model_pretrain/Qwen2.5-72B-Instruct-GPTQ-Int4"
output_path = '/root/ld/ld_project/MiniCPM-CookBook/mbti_role_play/mbti_sft_dpo_data/dpo_out_data/remake_sft_data2.json'
no_fix_path ='/root/ld/ld_project/MiniCPM-CookBook/mbti_role_play/mbti_sft_dpo_data/dpo_out_data/nofix_data.json'
batch_size = 8

import re
def get_qa_from_output(output):
    pattern1 = r'「修改后的答案」(.*?)「修改后的答案」'
    pattern2 = r'「修改后的问题」(.*?)「修改后的问题」'
    anwser = re.search(pattern2, output, re.DOTALL)
    question = re.search(pattern1, output,re.DOTALL)
    if anwser and question:
        return question.group(1), anwser.group(1)
    else:
        return None,None


import json
from vllm import LLM, SamplingParams
import argparse
import json
import pandas as pd
# 假设文件名为 'data.json'
with open('/root/ld/ld_project/MiniCPM-CookBook/mbti_role_play/mbti_sft_dpo_data/dpo_out_data/nofix_data.json', 'r') as file:
    data = json.load(file)

mbti_labels = []
prompts = []
for d in data:
    mbti = d['instruction'].split('\n')[0]
    mbti_labels.append(mbti)
    question = '\n'.join(d['instruction'].split('\n')[1:])
    answer = d['output']
    prompts.append(input_prompt.format(requirement, example, question, answer))


prompts = ["""<|im_start|> system\n you are a helpful assistant<|im_end|>\n<|im_start|> user\n {}<|im_end|>\n""".format(prompt) for prompt in prompts]
params_dict = {
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1,    
    "frequency_penalty": 1.0,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}
sampling_params = SamplingParams(**params_dict)

llm = LLM(model=model_path, tensor_parallel_size=8,max_model_len=2048, dtype='bfloat16',trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=0.8)
out_datas=[]
no_fix=[]
origin_index=0
for index in range(0, len(prompts), batch_size):
    outputs = llm.generate(prompts[index:index+batch_size], sampling_params)
    for i in range(len(outputs)):
        output=outputs[i].outputs[0].text
        answer,question=get_qa_from_output(output)
        if question and answer:
            input_line = {'instruction':'', 'input':mbti_labels[origin_index]+question,'output':answer}
            out_datas.append(input_line)
        else:
            input_line=data[origin_index]
            print(input_line)
            no_fix.append(input_line)        
        origin_index+=1

    if index%128==0:
        json_data = json.dumps(out_datas, ensure_ascii=False, indent=4)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(json_data)

        json_data = json.dumps(no_fix, ensure_ascii=False, indent=4)

        # 写入文件
        with open(no_fix_path, 'w', encoding='utf-8') as file:
            file.write(json_data)
json_data = json.dumps(out_datas, ensure_ascii=False, indent=4)

        # 写入文件
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(json_data)
    
json_data = json.dumps(no_fix, ensure_ascii=False, indent=4)

    # 写入文件
with open(no_fix_path, 'w', encoding='utf-8') as file:
    file.write(json_data)
