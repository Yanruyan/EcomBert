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
pip install accelerate
```
- sentencepiece
```
pip install sentencepiece
```
- 其他依赖
```
pip install tqdm scikit-learn
pip install tensorboard
pip install pandas
```

## Continue Pre-train
### 训练概况
cpt任务：mlm<br>
训练语料：200万电商商品标题、评论数据<br>
训练条件：4090单卡<br>
epoch：2<br>
训练时间：3小时<br>

### cpt后模型与原生模型效果对比
cpt模型相比原生模型，效果提升显著，如下：

| 模型版本             | Average MLM loss | Perplexity |
|------------------|----|--------|
| 原生roberta-base   | 4.8544 | 128.3075 |
| cpt-roberta-ecom | 2.0930 | 8.1090 |


## Fine-tune

