################################################################
#  使用电商数据，对roterta-base模型进行cpt训练，2卡4090，分布式训练
################################################################

from transformers import (
    RobertaTokenizerFast, RobertaForMaskedLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, LineByLineTextDataset
)

# 1. 路径和参数
model_name_or_path = "roberta-base"
train_file = "../data/sampled_train.txt"  # 每行一条文本
output_dir = "../roberta_ecommerce_ckpt"
per_device_train_batch_size = 64
num_train_epochs = 2

# 2. 加载分词器和模型（目录中存放model+tokenizer，微调的加载也一样）
tokenizer = RobertaTokenizerFast.from_pretrained(model_name_or_path)
model = RobertaForMaskedLM.from_pretrained(model_name_or_path)

# 3. 构建数据集
# 把每一行分词后，按照 block_size 的长度进行截断或填充（pad），从而生成每个样本
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)

# 4. MLM数据处理器
# 会在每个batch被送入模型前，自动、随机地对输入做mask（随机将一部分输入的token（通常为15%）进行mask），无需提前静态处理数据
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,   # MASK
    mlm_probability=0.15,   # MASK比例
)

# 5. 训练参数（自动支持多卡分布式，需用torchrun/accelerate/deepspeed启动）
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    save_steps=2000,
    save_total_limit=10,
    logging_steps=100,
    prediction_loss_only=True,
    fp16=True,  # 启用混合精度训练
    dataloader_num_workers=4,
    report_to="none",
)

# 6. Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 7. 开始训练
trainer.train()

# 保存最终模型（保存model+tokenizer）
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)