import os
import torch
import numpy as np
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, DataLoader
import deepspeed
from tqdm import tqdm

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        return self.lines[idx]


def eval_one_model(model_name_or_path, ds_config, test_file, batch_size, mlm_p, max_length):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
    model.eval()
    model = model.to(device)

    model, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=None,
        training_data=None
    )

    dataset = LineByLineTextDataset(test_file)
    def encode_fn(texts):
        return tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_p
    )
    def collate_fn(batch):
        enc = encode_fn(batch)
        enc_batch = []
        for i in range(enc['input_ids'].size(0)):
            item = {k: v[i] for k, v in enc.items()}
            enc_batch.append(item)
        collated = data_collator(enc_batch)
        return {k: v for k, v in collated.items()}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_name_or_path}", disable=local_rank != 0):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            losses.append(loss.detach().cpu().item())

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    if local_rank == 0:
        print(f"\n===> Model: {model_name_or_path}")
        print(f"      Average MLM loss: {avg_loss:.4f}")
        print(f"      Perplexity: {perplexity:.4f}")
    return avg_loss, perplexity

def main():
    deepspeed.init_distributed()
    # 评估原生roberta
    eval_one_model(
        model_name_or_path="roberta-base",
        ds_config="./ds_config_eval.json",
        test_file="../data/sampled_test.txt",
        batch_size=32,
        mlm_p=0.15,
        max_length=128
    )
    # 评估cpt后的roberta模型
    eval_one_model(
        model_name_or_path="../roberta_ecommerce_ckpt",
        ds_config="./ds_config_eval.json",
        test_file="../data/sampled_test.txt",
        batch_size=32,
        mlm_p=0.15,
        max_length=128
    )

if __name__ == "__main__":
    main()