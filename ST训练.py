import logging
import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from datasets import Dataset
from step_04_加载数据 import ContrastiveCommunityDataset

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logger

def main(csv_path: str,output_path: str):
    logger = configure_logging()
    logger.info("Loading model")
    model = SentenceTransformer("./model")
    print(model)
    print(model[0].auto_model)
    tokens = model.tokenizer("Hello world")
    print(tokens["input_ids"]) # 分词后添加了<|endoftext|> [9707, 1879, 151643]
    print("Padding side:", model.tokenizer.padding_side) # Padding side: right
    print("pad token:", model.tokenizer.pad_token) # pad token: <|endoftext|>

    # 加载你的自定义数据集
    csv_path = os.path.join(os.getcwd(), csv_path)
    custom_dataset = ContrastiveCommunityDataset(csv_path)
    # 转换为最新 SentenceTransformer Trainer 接受的HF三元组格式
    anchors, positives, negatives = [], [], []
    for ex in custom_dataset:
        anchors.append(ex.texts[0])
        positives.append(ex.texts[1])
        negatives.append(ex.texts[2])
    
    hf_dataset = Dataset.from_dict({
        "sentence_0": anchors,
        "sentence_1": positives,
        "sentence_2": negatives,
    })
    logger.info(f"Loaded dataset with {len(hf_dataset)} triplets")

    # 定义损失函数
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.3
    )

    # 配置训练参数
    output_path = os.path.join(os.getcwd(), output_path)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        warmup_steps=0,
        learning_rate=2e-5,
        logging_steps=20,               # 每10步打印一次 loss
        log_level="info",
        save_strategy="no",             # 如需保存中间模型可改为 "steps" 或 "epoch"
        report_to="none",               # 不使用 wandb/tensorboard
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(), # 自动启用混合精度（如果支持）
    )

    # 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=hf_dataset,
        loss=train_loss,
    )

    logger.info("Starting training")
    trainer.train()

    # 保存最终模型
    model.save(output_path)
    logger.info(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main("community_cleaned.csv","output_qwen3_st_finetuned")
