# EcomBert
使用跨境电商商品、评论数据，对bert模型进行继续预训练，使其学习到跨境电商场景专业知识、专有名词，在下游nlp任务上有更好的效果

## 开发环境
### 1、硬件环境
```
cuda版本：12.1
gpu：rtx 4090，<=8卡
内存：512g
```
### 2、python环境
- python 3.10（conda虚拟环境管理）
```
conda create -n bert python=3.10
source activate bert
```
### 3、安装依赖工具
- pytorch
```
建议用官网命令，选择CUDA 12.x与对应版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- DeepSpeed
```
pip install deepspeed
```
- Transformers、Datasets
```
pip install transformers datasets
```
- accelerate
```

```
- sentencepiece
- tqdm scikit-learn
- tensorboard
- pandas