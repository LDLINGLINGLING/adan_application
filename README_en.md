

# Interesting Projects

- [OCR_VG](#ocr_vg)
- [Cross-Modal Search Based on MiniCPMV2.0](#cross-modal-search-based-on-minicpmv20)
- [Complex Agent Project](#complex-agent-project)
- [MBTI Role Playing](#mbti-role-playing)
- [Hybrid Modality Fine-Tuning](#hybrid-modality-fine-tuning)
- [Running RAG with 4GB Memory](#running-rag-with-4gb-memory)
- [AWQ Quantization for MiniCPMV2.6](#awq-quantization-for-minicpmv26)
- [Cold Start to Acquire Agent Data](#cold-start-to-acquire-agent-data)

All of the above projects are original works. Feel free to use them, but please respect my intellectual property rights and give a star if you find them useful.

## OCR_VG

This project integrates OCR and localization tasks, considering layout issues. The project is located in the OCR_VG folder. You can access the [Text Recognition and Localization Tutorial](https://modelbest.feishu.cn/wiki/HLRiwNgKEic6cckGyGucFvxQnJw?from=from_copylink).

### Project Effects

<div align="center"> <img src="./OCR_VG/out/1.jpg" alt="Project Effect 1" width="500"/> <br /> <img src="./OCR_VG/out/4.jpg" alt="Project Effect 2" width="500"/> </div>

## Cross-Modal Search Based on MiniCPMV2.0

Using multi-vector and contrastive learning methods, the goal is to train an end-to-end cross-modal search model that can understand dense text and complex tables. [Model Link](https://www.modelscope.cn/models/linglingdan/Minicpmv_embeding_multi_vector)

### Demonstration of Results:

1. Input 20 candidate images:
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-2.png" alt="Candidate Images" width="800"/>
   </div>
2. Input query text for search:
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-1.png" alt="Query Text" width="800"/>
   </div>
3. Obtain the image most similar to the query.
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-3.png" alt="Most Similar Image" width="800"/>
   </div>
4. Experimental Results:
   300 validation set image-text pairs, with a Top1 match accuracy of 96%.

### Usage Tutorial

See the [Feishu Document](https://modelbest.feishu.cn/docx/CGEzdu25MoXkoVx3Qoac0e25nvg?from=from_copylink)

## Cold Start to Acquire Agent Data

To quickly build an Agent, I have developed a tool for generating agent training data using large models, saving you 95% of the time. This includes data generation in qwen (react) and minicpm formats.

[Zero Modification Data Generation Example](./agent_demo/react_qa_react.json)
[Generate Code Click Here](./agent_demo/get_react_data.py)

## Complex Agent Project

Using MiniCPMV2.6, I completed the project for the paper [AutoPlan](https://github.com/LDLINGLINGLING/AutoPlan), which can plan and execute complex tasks.

### Demonstration of Results:

1. Input query:
   <div align="center">
     <img src="./agent_auto_plan/asset/image.png" alt="Input Query" width="1000"/>
   </div>
2. Obtain task decomposition
   <div align="center">
     <img src="./agent_auto_plan/asset/image-1.png" alt="Task Decomposition" width="1000"/>
   </div>
3. Obtain task execution
   <div align="center">
     <img src="./agent_auto_plan/asset/image-2.png" alt="Task Execution 1" width="1000"/>
     <img src="./agent_auto_plan/asset/image-3.png" alt="Task Execution 2" width="1000"/>
   </div>
4. Obtain final answer
   <div align="center">
      <img src="./agent_auto_plan/asset/image-4.png" alt="Final Result" width="400"/>
   </div>

### Usage Tutorial

See the [Feishu Document](https://modelbest.feishu.cn/wiki/IgF0wRGJYizj4LkMyZvc7e2Inoe?from=from_copylink)

## MBTI Role Playing

Unlike the team at Peking University's Chatlaw, which trains a model for each personality, I have completed seamless switching between 16 personalities using a single 2B model (enabling role-playing of multiple personalities).

### Usage Tutorial

[Role Playing](https://modelbest.feishu.cn/docx/EcNjdGwvwoLkDrxpVrQcLwlknCg?from=from_copylink)

### Demonstration of Results

<div align="center">
  <img src="./mbti_role_play/demo_img/ESTP.PNG" alt="ESTP Personality" width="800"/>
  <img src="./mbti_role_play/demo_img/INTJ.PNG" alt="INTJ Personality" width="800"/>
  <img src="./mbti_role_play/demo_img/ESTP1.PNG" alt="ESTP Personality 1" width="800"/>
  <img src="./mbti_role_play/demo_img/INTJ1.PNG" alt="INTJ Personality 1" width="800"/>
</div>

## Hybrid Modality Fine-Tuning

The fine-tuning of MiniCPMV only opened up training for text-image dual modalities. This project modified the training mode to include both pure text and text-image pairs, located in the MIniCPM_Series_Tutorial/ft_language_replace_file folder.

### Usage Tutorial

You can access the [Hybrid Modality Fine-Tuning Tutorial](https://modelbest.feishu.cn/wiki/Y1NbwYijHiuiqvkSf0jcUOvFnTe?from=from_copylink)

The degradation of language modality capability due to alignment training refers to the multimodal model mllm, where the ability to respond to pure language inputs decreases, often referred to as the "alignment tax" (essentially another form of catastrophic forgetting). One simple method to mitigate catastrophic forgetting is to mix in raw data. For the loss of language capability in multimodal models, this involves mixing in language data. However, the question of which language data to mix and in what proportion is not the focus of this article, and I am also unable to solve this problem. For practical applications, mllm does not need to be a jack-of-all-trades in language capabilities; rather, it needs to maintain basic Q&A and specialized response capabilities in a specific domain while having excellent multimodal capabilities.

## Running RAG with 4GB Memory

<div align="center">
  <img src="./4G_memory_rag/image.png" alt="4GB Memory RAG 1" width="600"/>
  <img src="./4G_memory_rag/image1.png" alt="4GB Memory RAG 2" width="600"/>
</div>

There isn't much to explain here. This project allows running RAG with very low memory.

### Usage Tutorial

Tutorial available at [RAG](https://modelbest.feishu.cn/wiki/G5NlwYGGAiJWGmkCc4NcQ3sAnms?from=from_copylink)

## AWQ Quantization for MiniCPMV2.6

Since bnb quantization of MiniCPMV2.6 cannot be loaded by vllm, I adapted autoawq. I have already submitted a PR to autoawq, and once merged, it will be directly usable.

### Usage Tutorial
[Feishu Tutorial Link](https://modelbest.feishu.cn/wiki/PAsHw6N6xiEy0DkJWpJcIocRnz9?from=from_copylink)
Usage steps are as follows:

1. Get the personal autoawq branch
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoAWQ
   cd AutoAWQ
   git checkout minicpmv2.6
   pip install e .
   ```
2. Replace the `modeling_minicpmv.py` file in the `MiniCPM_Series_Tutorial/MiniCPMV2_6_awq` directory with the same-named file in the MiniCPMV2.6 model save path.
3. Modify the `model_path` in `MiniCPM_Series_Tutorial/MiniCPMV2_6_awq/quantize.py` to your MiniCPMV2.6 save path.
4. Run `quantize.py`.

After obtaining the awq quantized MiniCPMV2.6 model, you can deploy it using the original vllm, with the same deployment method. The model size is reduced from 16GB to 7GB of VRAM.
<div align="center">
  <img src="./MiniCPMV2_6_awq/image.png" alt="AWQ Quantization" width="500"/>
</div>
