
## AutoGPTQ

### 设备要求
至少存在一张的NVIDIA 20系以上显卡，需要12GB以上显存。

### 方法1: 直接获取量化后的GPTQ权重（推荐）
```bash
git clone https://huggingface.co/openbmb/MiniCPM3-4B-GPTQ-Int4
```

### 方法2: 自行量化（进行SFT后需要量化推荐）

1. **获取MiniCPM模型权重**
   ```bash
   git clone https://huggingface.co/openbmb/MiniCPM3-4B
   ```

2. **获取量化脚本**
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM
   ```

3. **安装AutoGPTQ分支**
   这里获取我fork的分支代码。（已经提了PR）
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoGPTQ.git
   cd AutoGPTQ
   git checkout minicpm3
   pip install -e .
   ```

4. **开始量化**
   ```bash
   cd MiniCPM/quantize
   # 以下代码中no_quantized_path修改为MiniCPM3权重保存的地址，save_path3为量化模型保存的地址
   python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
   ```
