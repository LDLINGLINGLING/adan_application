from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '900d38ea-1d06-4620-9cfe-23a8e8a137d5'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="linglingdan/mbti_role_play", 
    model_dir="/root/ld/ld_project/pull_request/LLaMA-Factory/saves/minicpm3_1b/dpo/checkpoint-367" # 本地模型目录，要求目录中必须包含configuration.json
)