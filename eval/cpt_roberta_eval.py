####################################################################################
#  对比cpt后的roberta与原生roberta的基础能力的对比。
#  通常有2个指标：
#  （1）困惑度（PPL，模型对masked prediction的平均损失，交叉熵loss）
#  （2）MASK预测的准确率，在测试集上随机遮盖15%的Token，计算预测正确的比例，计算公式如下：
#        accuracy = 预测正确的mask的token数/mask的总的token数
####################################################################################

import torch
from transformers import (
    RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling
)


# 环境
device = torch.device("cpu")

# 路径
origin_model = 'roberta-base'
finetuned_model = '../roberta_ecommerce_ckpt'
test_file = '../data/sampled_test.txt'

# 加载原生roberta和cpt后的roberta
tokenizer = RobertaTokenizerFast.from_pretrained(origin_model)
model_origin = RobertaForMaskedLM.from_pretrained(origin_model).eval()
model_ft = RobertaForMaskedLM.from_pretrained(finetuned_model).eval()
model_origin = model_origin.to(device)
model_ft = model_ft.to(device)

# 计算困惑度PPL
def compute_mlm_loss(model, tokenizer, test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 对每行文本单独编码
    inputs = [tokenizer(line, return_tensors='pt', truncation=True, max_length=128) for line in lines]

    # 准备data_collator需要的格式
    examples = [{'input_ids': input['input_ids'][0]} for input in inputs]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    batch = data_collator(examples)

    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss.item()
    return loss


print("原生模型PPL:", compute_mlm_loss(model_origin, tokenizer, test_file))
print("继续预训练后模型PPL:", compute_mlm_loss(model_ft, tokenizer, test_file))
