#!/usr/bin/env python
# encoding: utf-8
import re
from transformers import AutoTokenizer,AutoModelForCausalLM
from .cli_demo import get_relevant_image
import requests
import json
import base64

def get_delivery_date(order_id=None):
    if order_id == None:
        return "没有订单号无法查询"
    else:
        print("get_delivery_date:这里应该替换查询方法函数，结果用return返回")
        return "2024-09-02"
def text_to_image_search(text_description,images_path):
    if text_description != None and images_path != None:
        return get_relevant_image(text_description,images_path)
def image_gen_text(image_path,query):
    with open(image_path, 'rb') as image_file:
        # 将图片文件转换为 base64 编码
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
    "model": "minicpm-v",
    "prompt": query,
    "stream": False,
    "images": [encoded_string]
    }

    # 设置请求 URL
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)

    return response.json()["response"]
    
def fake_tool_execute(toolcalls):
    tool_name = toolcalls['function']['name']
    arguments = toolcalls['function']["arguments"]
    if tool_name == "text_to_image_search":
        return text_to_image_search(arguments["text_description"],arguments["images_path"])
    elif tool_name == "image_gen_text":
        return image_gen_text(arguments["image_path"],arguments["query"])
    else:
        return "没有这个工具"
def get_response_call(tool_call_str):


    # 正则表达式
    pattern = r'(?<=```python\n)(.*?)(?=\n```\n)'

    # 使用正则表达式匹配
    match = re.search(pattern, tool_call_str)

    if match:
        function_call = match.group(1)
        return function_call
    else:
        return None
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "text_to_image_search",
            "description": "According to the input language description, find the most relevant picture with the language description under the specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_description": {
                        "type": "string",
                        "description": "input language description of the target image",
                    },
                    "images_path": {
                        "type": "string",
                        "description": "Contains the path to where the target image is",
                    },
                },
                "required": ["text_description","images_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_gen_text",
            "description": "According to the input language description, find the most relevant picture with the language description under the specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the target image",
                    },
                    "query": {
                        "type": "string",
                        "description": "Questions based on the target image",
                    },
                },
                "required": ["image_path","query"],
                "additionalProperties": False,
            },
        },
    }

]
query="你帮我找出/root/ld/ld_project/pull_request路径下有关股票走势的图片，然后告诉我最近半个月是上升还是下降。"
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    }
    ,
    {"role":"user",
     "content":"你帮我找出/root/ld/ld_project/pull_request路径下有关股票走势的图片，然后看图告诉我最近半个月是上升还是下降。"}
    
]
tokenizer = AutoTokenizer.from_pretrained(
    "/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True
)

# messages=[{
#         "role": "system",
#         "content": prompt,
#     }]
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/MiniCPM3-4B-GPTQ-Int4", trust_remote_code=True).cuda()
# while True:
#     prompt = tokenizer.apply_chat_template(
#     messages, tools=tools, tokenize=False, add_generation_prompt=True
#     )
#     prompt=tokenizer(prompt, return_tensors="pt")
#     response=model.generate(prompt.input_ids.cuda(),max_length=4096)
    
#     response=tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     print(response)
#     call_str=get_response_call(response)
#     tool_msg = {
#                     "role": "tool",
#                     "content": call_str,
#                     "tool_call_id": [random.randint(1, 100000000)],
#                 }
#     messages.append(tool_msg)
#     print(call_str)
#     print(eval(call_str))

while True:
    if model not in locals():
        model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/MiniCPM3-4B-GPTQ-Int4", trust_remote_code=True).cuda()

    #temp = copy.deepcopy(messages[-1])
    prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt=tokenizer(prompt, return_tensors="pt")
    response=model.generate(prompt.input_ids.cuda(),max_length=4096,do_sample=False)
    response = response[:, prompt.input_ids.shape[1]:]
    response=tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    #messages[-1]=temp
    #response = outputs[0].outputs[0].text
    msg = tokenizer.decode_function_call(response)
    if (
        "tool_calls" in msg
        and msg["tool_calls"] is not None
        and len(msg["tool_calls"]) > 0
    ):
        messages.append(msg)
        #print(msg)
        for toolcall in msg["tool_calls"]:
            del model
            tool_response = fake_tool_execute(toolcall)
            tool_msg = {
                "role": "tool",
                "content": tool_response,
                "tool_call_id": toolcall["id"],
            }
            messages.append(tool_msg)
            #print(tool_msg)
    else:
        messages.append(msg)
        #print(msg)
        break

