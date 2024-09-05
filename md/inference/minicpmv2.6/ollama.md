
# Ollama 推理

## 设备要求

- 运行非量化版本需要19GB以上内存
- 运行量化版本需要8GB以上内存

## 步骤1：获取gguf模型

按照上述Llama.cpp教程获取gguf模型。语言模型最好是量化模型。

## 步骤2：安装依赖包

使用Homebrew安装依赖包：

```sh
brew install ffmpeg
brew install pkg-config
```

## 步骤3：获取OpenBMB官方Ollama分支

```sh
git clone -b minicpm-v2.6 https://github.com/OpenBMB/ollama.git
cd ollama/llm
git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
cd ../
```

## 步骤4：环境需求

- cmake version 3.24 or above
- go version 1.22 or above
- gcc version 11.4.0 or above

使用Homebrew安装所需工具：

```sh
brew install go cmake gcc
```

## 步骤5：安装大模型依赖项

```sh
go generate ./...
```

## 步骤6：编译Ollama

```sh
go build .
```

## 步骤7：启动Ollama服务

编译成功后，在Ollama主路径启动Ollama：

```sh
./ollama serve
```

## 步骤8：创建一个ModelFile

编辑ModelFile：

```sh
vim minicpmv2_6.Modelfile
```

ModelFile的内容如下：

```plaintext
FROM ./MiniCPM-V-2_6/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-2_6/mmproj-model-f16.gguf

TEMPLATE """{{ if .System }}<|im_start|>system

{{ .System }}<|im_end|>{{ end }}

{{ if .Prompt }}<|im_start|>user

{{ .Prompt }}<|im_end|>{{ end }}

<|im_start|>assistant<|im_end|>

{{ .Response }}<|im_end|>"""

PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 2048
```
参数说明:

| first from | second from | num_keep | num_ctx |
|-----|-----|-----|-----|
| Your language gguf Model path | Your visionGguf modelpath | max connection limit| Max Model length |

9. 创建ollama模型实例：
```bash
ollama create minicpm2.6 -f minicpmv2_6.Modelfile
```

10. 另起一个命令行窗口，运行ollama模型实例：
```bash
ollama run minicpm2.6
```

11. 输入问题和图片 URL，以空格分隔
```bash
What is described in this picture? /Users/liudan/Desktop/11.jpg
```