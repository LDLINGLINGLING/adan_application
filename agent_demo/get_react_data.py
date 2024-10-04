import re
from vllm import LLM, SamplingParams
from build_react_prompt import build_input_text,TOOL_DESC,PROMPT_REACT,parse_latest_plugin_call
import json
import json5
#from agent_demo import *
tools = [
            {
            'name_for_human': '图生文',
            'name_for_model': 'image_gen_prompt',
            'excute_function': False,
            'description_for_model': '图生文是一个可以看图生成文字描述的服务，输入一张图片的地址，将返回图片详细逼真的表述',
            'example':'帮我看一下www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png这张图片上的今日股价是多少',
            'parameters': [
                {
                    'name': 'image_path',
                    'description': '需要图片描述的URL或者本地地址',
                    'scope':None,
                    'required': True,
                    'schema': {'type': 'string'},
                }
            
            ]
        },
        {
            'name_for_human': '知识图谱',
            'name_for_model': 'knowledge_graph',
            'excute_function': True,
            'description_for_model': '知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性',
            'example':'帮我查一下敌方直升机的续航里程',
            'parameters': [
                {
                    'name': 'weapon_query',
                    'description': '武器名称',
                    'scope':['直升机','坦克','反坦克导弹','直升机','火箭炮','所有武器'],
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'name': 'attribute',
                    'description': '武器的属性',
                    'scope':['射程','续航里程','重量','速度','承载量','适应场景','克制武器'],
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        }
        ]
def function_call(plugin_name, plugin_args):
    args_dict = json5.loads(plugin_args)
    if plugin_name == 'knowledge_graph':
        weapon_name = args_dict['weapon_query']
        attribute = args_dict['attribute']
        kg={'直升机':{'飞行高度':'0.3km以内','携带武器':'火箭弹','克制武器':'对空导弹','重量':'3000kg',"速度":"100km/h","射程":"2km",'适应场景':'空战','续航':'500km','满载人数':'7人','承载重量':'10000kg','续航里程':'1000km'},
                        '反坦克导弹':{'重量':'100kg','射程':'0.5千米','克制武器':'拦截导弹','适应场景':'打击重装甲武器','速度':'200km/h'},
                        '步兵':{'射程':'0.3km','克制武器':'无人机','适应场景':'陆地',"速度":'40km/h','重量':'60kg','承载重量':'50kg'},
                        '无人机':{'速度':'100km/h','重量':'10kg','适应场景':'侦察和暗杀','飞行高度':'0.3km以下','克制武器':'电磁攻击','续航':'50km'},
                        '豹2A7坦克':{'速度':'50km/h','携带武器':'激光炮','克制武器':'反坦克导弹',"射程":"5km",'重量':'10000kg','续航':'1000km','承载重量':'200000kg','满载人数':'5人' ,'适应场景' :'野战和掩护步兵'},
                        '黑狐坦克':{'速度':'70km/h','携带武器':'主炮','克制武器':'反坦克导弹',"射程":"15km",'重量':'10000kg','承载重量':'50000kg','续航':'1000km','满载人数':'5人','适应场景' :'野战和掩护步兵'},
                        "火箭炮":{'速度':'4500km/h','重量':'500kg','射程':'1000km','适应场景':'超远程打击','飞行高度':'万米高空','克制武器':'拦截导弹'},
                        "雷达":{'重量':'5000kg','探测范围':'2km以上20km以下','适应场景':'探测敌军'},
                        '装甲车':{'速度':'80km/h','携带武器':'副炮','克制武器':'穿甲弹',"射程":"0.5km",'重量':'10000kg','承载重量':'10000kg','续航':'600km','满载人数':'10人'},
                        '狙击枪':{'射程':'1.2km','重量':'30kg','适应场景':'暗杀'}
                        }
        if weapon_name != '所有武器':
            try:
                return '{}的{}是:{}'.format(weapon_name,attribute,kg[weapon_name][attribute])
            except:
                if weapon_name not in kg:
                    return '该武器不存在'
                else:
                    return '{}的{}属性不存在'.format(weapon_name,attribute)
        return kg

gen_batch = 5
model_path = "/root/ld/ld_model_pretrained/Qwen2.5-72B-Instruct-GPTQ-Int4"
save_question_json = '/root/ld/ld_project/pull_request/MiniCPM_Series_Tutorial/agent_demo/question_react.json'
save_react_qa_json = '/root/ld/ld_project/pull_request/MiniCPM_Series_Tutorial/agent_demo/react_qa_react.json'
inference_batch_size = 8



def get_answer_from_output(output):
    pattern = r'「问题开始」(.*?)「问题结束」'
    questions = re.findall(pattern, output, re.DOTALL)
    questions = [q.strip() for q in questions]
    return questions


def get_tool_description(tool):
    tool_descp = "工具名称是{},作用是{},".format(tool['name_for_human'],tool['description_for_model'])
    for t in tool['parameters']:
        if t['required']:
            if t['scope']:
                tool_descp+='参数:{}是必须输入的，作用是{},该参数的取值范围是{}。'.format(t['name'], t['description'], t['scope'])
            else:
                tool_descp+='参数:{}是必须输入的，作用是{}。'.format(t['name'], t['description'])
        elif t['scope']:
            tool_descp+='参数:{}是可选的，作用是{},该参数的取值范围是{}。'.format(t['name'], t['description'], t['scope'])
        else:
            tool_descp+='参数:{}是可选的，作用是{}。'.format(t['name'], t['description'])
    return tool_descp
prompt_template = """你是一个智能助手，现在我请你为以下工具生成问题，要求生成的问题能够被这个工具解决。工具的详细介绍如下：\n{}\n我现在给你一个关于此工具问题的示例「问题开始」{}「问题结束」,接下来请你根据此示例和工具描述再生成{}个能够使用该工具解决的问题，并且用「问题开始」和「问题结束」将其包裹。"""


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

llm = LLM(model=model_path, tensor_parallel_size=8,max_model_len=4096, dtype='bfloat16',trust_remote_code=True,enforce_eager=True,gpu_memory_utilization=0.8)


all_questions = []
all_react_prompt = []
questinos_dict = {}
for tool in tools:
    questions = []
    while True:
        tool_description = get_tool_description(tool)
        input_prompt = prompt_template.format(tool_description,tool['example'],10)
        input_prompt = """<|im_start|> system\n you are a helpful assistant<|im_end|>\n<|im_start|> user\n {}<|im_end|>\n""".format(input_prompt)
        outputs = llm.generate(input_prompt, sampling_params)
        output=outputs[0].outputs[0].text
        questions.extend(get_answer_from_output(output))
        
        if len(questions)>=gen_batch:
            all_questions.extend(questions)
            print(questions)
            questinos_dict[tool['name_for_model']] = questions
            break
with open(save_question_json, 'w', encoding='utf-8') as f:
    json.dump(questinos_dict, f, ensure_ascii=False, indent=4)
    print('{}条输入指令已经保存到{}'.format(len(all_questions),save_question_json))
    

react_question = [build_input_text([(q,'')],tools) for q in all_questions]
params_dict["top_k"] = 1
params_dict['stop'] = ['Observation:']

react_qa = []
sampling_params = SamplingParams(**params_dict)
for index in range(0, len(react_question), inference_batch_size):
    outputs = llm.generate(react_question[index:index+inference_batch_size], sampling_params)
    for i in range(len(outputs)):
        output=outputs[i].outputs[0].text
        try:
            plugin_name, plugin_args, text = parse_latest_plugin_call(output)
            excute_flag = True 
            for tool in tools:
                if tool['name_for_model'] == plugin_name and tool['excute_function']==False:
                    excute_flag = False
                    second_input = react_question[index+i]+output+'Observation: '
                    output2 = llm.generate(second_input, sampling_params)[0].outputs[0].text
            if excute_flag:
                observation = function_call(plugin_name,plugin_args)
                second_input = react_question[index+i]+output+'Observation: {}'.format(observation)
            output2 = llm.generate(second_input, sampling_params)[0].outputs[0].text
            print(output2)
            #react_qa.append({react_question[index+i]: second_input[len(react_question[index+i]):]+output2})
            react_qa.append({'instruction':"You are a helpful assistant.",'input':react_question[index+i][75:-33],'output':second_input[len(react_question[index+i]):]+output2})
        except:
            pass
        
        
with open(save_react_qa_json, 'w', encoding='utf-8') as f:
    json.dump(react_qa, f, ensure_ascii=False, indent=4)
    print('{}条react qa数据已经保存到{}'.format(len(react_qa),save_react_qa_json))
