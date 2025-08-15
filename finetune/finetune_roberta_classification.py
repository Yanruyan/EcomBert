import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import deepspeed
from argparse import ArgumentParser


class TextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    text, label = line.strip().split('\t')
                    self.samples.append((text, int(label)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0
    print(f"Eval Accuracy: {acc:.4f}")
    return acc


def main():
    # 训练时一些参数
    # warmup参数：warmup step = 总训练step数 * prob（prob ~ [0.03~0.1]），官方默认0.1
    # 学习率参数：如果数据量较小或对 overfitting 敏感，建议用 1e-5 或 2e-5；如果数据量较大
    #           或模型收敛慢，可以适当提高至 3e-5 或 5e-5。这里我们设置为2e-5
    # weight_decay参数：0.01 是 AdamW 优化器的默认推荐值，适合大部分 NLP 微调任务；
    #           若模型有明显过拟合，可适当增加至 0.05 或 0.1，这里我们设置为0.01
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int, required=True, default=0)
    parser.add_argument('--train_file', type=str, required=True, default="../data/intention_train.txt")
    parser.add_argument('--test_file', type=str, required=True, default="../data/intention_test.txt")
    parser.add_argument('--model_name_or_path', type=str, required=True, default="../roberta_ecommerce_ckpt")
    parser.add_argument('--output_dir', type=str, required=True, default="../roberta_intention_classification_ckpt")
    parser.add_argument('--deepspeed', type=str, required=True, default="./finetune_ds_config.json")
    parser.add_argument('--epochs', type=int, required=True, default=3)
    parser.add_argument('--batch_size', type=int, required=True, default=32)
    parser.add_argument('--max_length', type=int, required=True, default=128)
    parser.add_argument('--learning_rate', type=float, required=True, default=2e-5)
    parser.add_argument('--weight_decay', type=float, required=True, default=0.01)
    parser.add_argument('--warmup_steps', type=int, required=True, default=280)
    args = parser.parse_args()

    # 初始化DeepSpeed分布式环境（即使只有1个GPU，也需要这样做）
    deepspeed.init_distributed()
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    # 训练与测试集
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = TextClassificationDataset(args.train_file, tokenizer, args.max_length)
    test_dataset = TextClassificationDataset(args.test_file, tokenizer, args.max_length)

    # PyTorch的DataLoader，自动分batch
    # DataLoader遍历得到：list[batch_sample]，batch_sample的shape=(batch_size, sample_dim)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 加载cpt后的roberta模型，用于文本2分类
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    model = model.to(device)

    # AdamW优化器：BERT/Roberta常用优化器，带权重衰减
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 训练初期使用"热身"(warmup)策略，然后线性衰减学习率。这种调度策略在许多自然语言处理任务中表现良好
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,     # warmup的step数
        num_training_steps=total_steps          # 总step数
    )

    # 将PyTorch模型转换为支持DeepSpeed优化的模型，并配置分布式训练环境
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=scheduler,
        config=args.deepspeed
    )

    # 开始训练
    best_acc = 0
    for epoch in range(args.epochs):
        model_engine.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            pbar.set_postfix({'loss': loss.item()})

        # 每个epoch训练完，计算1次评估指标：准确率Acc
        # 将准确率最高的模型，保存到输出目录下
        if local_rank == 0:
            print(f"Running evaluation after epoch {epoch+1}")
            acc = evaluate(model_engine.module, test_dataloader, device)
            if acc > best_acc:
                best_acc = acc
                model_engine.module.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"New best model saved to {args.output_dir} (acc={acc:.4f})")


if __name__ == "__main__":
    main()
