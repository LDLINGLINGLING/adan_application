# 有趣的项目

- [OCR_VG](#ocr_vg)
- [基于MiniCPMV2.0的跨模态搜索](#基于minicpmv20的跨模态搜索)
- [复杂agent项目](#复杂agent项目)
- [MBTI角色扮演](#mbti角色扮演)
- [混合模态微调](#混合模态微调)
- [4G显存玩转RAG](#4g显存玩转rag)
- [MiniCPMV2.6的AWQ量化](#minicpmv26的awq量化)

以下项目都是个人原创，如果需要可自取，但是注意保护我的个人知识产权，用了给个星星。

## OCR_VG

同时将OCR和定位任务融合，考虑排版问题，该项目在OCR_VG的文件夹下，在可以自取[文字识别与定位教程](https://modelbest.feishu.cn/wiki/HLRiwNgKEic6cckGyGucFvxQnJw?from=from_copylink)。

### 项目效果

<div align="center"> <img src="./OCR_VG/out/1.jpg" alt="项目效果1" width="500"/> <br /> <img src="./OCR_VG/out/4.jpg" alt="项目效果2" width="500"/> </div>

## 基于MiniCPMV2.0的跨模态搜索

使用多向量和对比学习的方法，目标是训练一个跨模态端到端搜索的模型，可以理解密集文字、复杂表格。[模型地址](https://www.modelscope.cn/models/linglingdan/Minicpmv_embeding_multi_vector)

### 效果展示：

1. 输入20张待选图片：
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-2.png" alt="待选图片" width="500"/>
   </div>
2. 输入query文字进行搜索:
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-1.png" alt="查询文字" width="500"/>
   </div>
3. 得到与query最相近的图片。
   <div align="center">
     <img src="./OCR_Multimodal_Search/asset/image-3.png" alt="最相近图片" width="500"/>
   </div>
4. 实验结果：
   300条验证集图文对，Top1匹配正确率高达96%。
   
### 使用教程

见[飞书文档](https://modelbest.feishu.cn/docx/CGEzdu25MoXkoVx3Qoac0e25nvg?from=from_copylink)

## 复杂agent项目

使用minicpmv2.6完成了论文[AutoPlan](https://github.com/LDLINGLINGLING/AutoPlan)的项目，能够对复杂任务进行规划和执行。

### 效果展示：

1. 输入query:
   <div align="center">
     <img src="./agent_auto_plan/asset/image.png" alt="输入query" width="500"/>
   </div>
2. 获得任务分解
   <div align="center">
     <img src="./agent_auto_plan/asset/image-1.png" alt="任务分解" width="500"/>
   </div>
3. 获得任务执行
   <div align="center">
     <img src="./agent_auto_plan/asset/image-2.png" alt="任务执行1" width="500"/>
     <img src="./agent_auto_plan/asset/image-3.png" alt="任务执行2" width="500"/>
   </div>
4. 获得最终答案
   <div align="center">
      <img src="./agent_auto_plan/asset/image-4.png" alt="最终结果" width="500"/>
   </div>

### 使用教程

见[飞书文档](https://modelbest.feishu.cn/wiki/IgF0wRGJYizj4LkMyZvc7e2Inoe?from=from_copylink)

## MBTI角色扮演

与北大Chatlaw团队每个人格训练一个模型不同，仅使用一个2b模型完成了16种人格的无缝切换（可玩人格分裂）

### 使用教程

[角色扮演](https://modelbest.feishu.cn/docx/EcNjdGwvwoLkDrxpVrQcLwlknCg?from=from_copylink)

### 效果展示

<div align="center">
  <img src="./mbti_role_play/demo_img/ESTP.PNG" alt="ESTP人格" width="500"/>
  <img src="./mbti_role_play/demo_img/INTJ.PNG" alt="INTJ人格" width="500"/>
  <img src="./mbti_role_play/demo_img/ESTP1.PNG" alt="ESTP人格1" width="500"/>
  <img src="./mbti_role_play/demo_img/INTJ1.PNG" alt="INTJ人格1" width="500"/>
</div>

## 混合模态微调

MiniCPMV的微调仅仅开放了图文双模态的训练，本项目修改了纯文本和图文对的混合训练模式，放在了MIniCPM_Series_Tutorial/ft_language_replace_file文件夹下，

### 使用教程

可以自取[混合模态微调教程](https://modelbest.feishu.cn/wiki/Y1NbwYijHiuiqvkSf0jcUOvFnTe?from=from_copylink)

对于对齐训练导致的语言模态能力下降是指的对齐后的多模态模型mllm，对于纯语言输入的回复能力有所下降，俗称对齐税（本质上也许是另外一种灾难性遗忘）。对于抑制灾难性遗忘一种比较简单的方法是混入原始数据，对于多模态的语言能力丢失，则是混入语言数据。这就迎来了另外一个问题，混入哪些语言数据，占比又是多少，这不是本文的重点，笔者亦无力解决这个问题。但是对于应用来说，mllm并不需要十项全能的语言能力，更多的是在有优秀的多模态能力下保持基础问答以及某一个领域的专业的回复能力。

## 4G显存玩转RAG

<div align="center">
  <img src="./4G_memory_rag/image.png" alt="4G显存RAG1" width="500"/>
  <img src="./4G_memory_rag/image1.png" alt="4G显存RAG2" width="500"/>
</div>

这个没什么好解释的，可以在极低显存下运行RAG，

### 使用教程

教程自取[RAG](https://modelbest.feishu.cn/wiki/G5NlwYGGAiJWGmkCc4NcQ3sAnms?from=from_copylink)

## MiniCPMV2.6的AWQ量化

由于bnb量化的minicpmv2.6无法用vllm加载，因此适配了autoawq，目前已经向autoawq提了pr，等合并后可以直接使用。

### 使用教程

使用方法如下：

1. 获取个人autoawq分支
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoAWQ
   cd AutoAWQ
   pip install e .
   ```
2. 将MiniCPM_Series_Tutorial/MiniCPMV2_6_awq/modeling_minicpmv.py文件替换掉minicpmv2.6模型保存路径下的同名文件
3. 修改MiniCPM_Series_Tutorial/MiniCPMV2_6_awq/quantize.py中的model_path为你minicpmv2.6的保存路径。
4. 运行quantize.py

获得minicpmv2.6的awq模型后可以使用原来的vllm进行部署，部署方式完全相同,模型从16g显存将为7g显存
<div align="center">
  <img src="./MiniCPMV2_6_awq/image.png" alt="AWQ量化" width="500"/>
</div>
