# MiniCPM-CookBook
 This is the introductory book of the minicpm series. Minicpm is the most powerful and efficient end-side model of openbmb open source.

# 目录和内容
## 模型介绍(✅)
### 语言模型的介绍与榜单评分图
### 多模态模型的介绍与榜单评分图
MiniCPM-V 2.6 是 MiniCPM-V 系列中最新、性能最佳的模型。该模型基于 SigLip-400M 和 Qwen2-7B 构建，共 8B 参数。与 MiniCPM-Llama3-V 2.5 相比，MiniCPM-V 2.6 性能提升显著，并引入了多图和视频理解的新功能。
![alt text](./asset/radar_2.6.png)
## 技术报告(✅)
- [语言模型技术报告](https://openbmb.vercel.app/?category=Chinese+Blog)
- [多模态模型技术报告](https://arxiv.org/abs/2408.01800)
## 支持硬件（云端、边端）(✅)
- Gpu
- Cpu
- Npu
- android
- mac
## 模型地址与下载(✅)
- [MiniCPM2.0](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- [MiniCPM3.0](https://huggingface.co/openbmb/MiniCPM3-4B)
- [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
- [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
## 推理部署(✅)
#### MiniCPM2.0
-  [MiniCPM_transformers_cuda](./md/inference/minicpm2.0/transformers.md)
-  [MiniCPM_vllm_cuda](./md/inference/minicpm2.0/vllm.md)
-  [MiniCPM__mlx_mac](./md/inference/minicpm2.0/mlx.md)
-  [MiniCPM_ollama_cuda_cpu_mac](./md/inference/minicpm2.0/ollama.md)
-  [MiniCPM_llamacpp_cuda_cpu](./md/inference/minicpm2.0/llama.cpp_pc.md)
-  [MiniCPM_llamacpp_android](./md/inference/minicpm2.0/llama.cpp_android.md)
-  [MiniCPM-S_powerinfer_cuda](./md/inference/minicpm2.0/powerinfer_pc.md)
-  [MiniCPM-S_powerinfer_android](./md/inference/minicpm2.0/powerinfer_android.md)
-  FAQ
#### MiniCPM3.0
-  [MiniCPM3.0_vllm_cuda](./md/inference/minicpm3.0/vllm.md)
-  [MiniCPM3.0_transformers_cuda_cpu](./md/inference/minicpm3.0/transformers.md)
#### MiniCPMV2.5
- [MiniCPM-Llama3-V 2.5_vllm_cuda](./md/inference/minicpmv2.5/vllm.md)
- [MiniCPM-Llama3-V 2.5_LMdeploy_cuda](./md/inference/minicpmv2.5/LMdeploy.md)
- [MiniCPM-Llama3-V 2.5_llamacpp_cuda_cpu](./md/inference/minicpmv2.5/llamacpp_pc.md)
- [MiniCPM-Llama3-V 2.5_ollama_cuda_cpu](./md/inference/minicpmv2.5/ollama.md)
- [MiniCPM-Llama3-V 2.5_transformers_cuda](./md/inference/minicpmv2.5/transformers_multi_gpu.md)
- [MiniCPM-Llama3-V 2.5_xinference_cuda](./md/inference/minicpmv2.5/xinference.md)
- [MiniCPM-Llama3-V 2.5_swift_cuda](./md/inference/minicpmv2.5/swift_python.md)
#### MiniCPMV2.6
- [MiniCPM-V 2.6_vllm_cuda](./md/inference/minicpmv2.6/vllm.md)
- [MiniCPM-V 2.6_vllm_api_server_cuda](./md/inference/minicpmv2.6/vllm_api_server.md)
- [MiniCPM-V 2.6_llamacpp_cuda_cpu](./md/inference/minicpmv2.6/llamacpp.md)
- [MiniCPM-V 2.6_transformers_cuda](./md/inference/minicpmv2.6/transformers_mult_gpu.md)
- [MiniCPM-V 2.6_swift_cuda](https://github.com/modelscope/ms-swift/issues/1613)
- FAQ
## 微调(✅)
#### MiniCPM2.0
- [MiniCPM_官方代码_sft_cuda](./md/finetune/minicpm2.0/sft.md)
- [MiniCPM_mlx_sft_lora_mac](./md/finetune/minicpm2.0/mlx_sft.md)
- [MiniCPM_llamafactory_RLHF_cuda](./md/finetune/minicpm2.0/llama_factory.md)
- FAQ
#### MiniCPMV2.5
- [MiniCPM-Llama3-V 2.5_官方代码_cuda](./md/finetune/minicpmv2.5/sft.md)
- [MiniCPM-Llama3-V-2_5_swift_cuda](./md/finetune/minicpmv2.5/swift.md)
- [混合模态训练](https://modelbest.feishu.cn/wiki/Y1NbwYijHiuiqvkSf0jcUOvFnTe?from=from_copylink)
#### MiniCPMV2.6
- [MiniCPM-V 2.6_官方代码_sft_cuda](./md/finetune/minicpmv2.6/sft.md)
- [MiniCPM-V 2.6_swift_sft_cuda](https://github.com/modelscope/ms-swift/issues/1613)
- [混合模态训练](https://modelbest.feishu.cn/wiki/As5Ow99z3i4hrCkooRIcz79Zn2f?from=from_copylink) 
- FAQ
## 模型量化(✅)
#### MiniCPM2.0
- [MiniCPM_awq量化](./md/quantize/minicpm2.0/awq.md)
- [MiniCPM_gguf量化](./md/inference/minicpm2.0/llama.cpp_pc.md)
- [MiniCPM_gptq量化](./md/quantize/minicpm2.0/gptq.md)
- [MiniCPM_bnb量化](./md/quantize/minicpm2.0/bnb.md)
#### MiniCPMV2.5
- [MiniCPM-Llama3-V 2.5bnb量化](./md/quantize/minicpmv2.5/bnb.md)
- [MiniCPM-Llama3-V 2.5gguf量化](./md/inference/minicpmv2.5/llamacpp_pc.md)
#### MiniCPMV2.6
- [MiniCPM-V 2.6_bnb量化](./md/quantize/minicpmv2.6/bnb.md)
- [MiniCPM-V 2.6_awq量化](./md/quantize/minicpmv2.6/awq.md)
- [MiniCPM-V 2.6_gguf量化](./md/inference/minicpmv2.6/llamacpp.md)
## 集成(✅)
- [langchain](./md/integrate/langchain.md)
- [openai_api](./md/integrate/opeai_api.md)
- chat_bot gradio
## 应用(✅)
### 语言模型
- [4G显存玩转rag_langchain](https://modelbest.feishu.cn/wiki/G5NlwYGGAiJWGmkCc4NcQ3sAnms?from=from_copylink) 
- [RLHF可控文本生成](https://modelbest.feishu.cn/wiki/ZEzGwgDgSi2Nk1kjAfFcrZn4nKd?from=from_copylink)
### 多模态模型
- [跨模态高清检索](https://modelbest.feishu.cn/wiki/NdEjwo0hxilCIikN6RycOKp0nYf?from=from_copylink)
- [文字识别与定位](https://modelbest.feishu.cn/wiki/HLRiwNgKEic6cckGyGucFvxQnJw?from=from_copylink)
- [Agent入门](https://modelbest.feishu.cn/wiki/HKQdwbUUgiL0HNkSetjctaMcnrw?from=from_copylink)
- [长链条Agent如何构造](https://modelbest.feishu.cn/wiki/IgF0wRGJYizj4LkMyZvc7e2Inoe?from=from_copylink)
- [多模态文档RAG](https://modelbest.feishu.cn/wiki/NwhIwkJZYiHOPSkzwPUcq6hanif?from=from_copylink)
## 开源社区合作(✅)
- [xtuner](https://github.com/InternLM/xtuner): [MiniCPM高效率微调的不二选择](https://modelbest.feishu.cn/wiki/AIU3wbREcirOm9kkvd7cxujFnMb#AMdXdzz8qoadZhxU4EucELWznzd)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)：[MiniCPM微调一键式解决方案](https://modelbest.feishu.cn/wiki/AIU3wbREcirOm9kkvd7cxujFnMb#BAWrdSjXuoFvX4xuIuzc8Amln5E)
- [ChatLLM框架](https://github.com/foldl/chatllm.cpp)：[在CPU上跑MiniCPM](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16/discussions/2#65c59c4f27b8c11e43fc8796)
## 开源协议与商务合作(✅)