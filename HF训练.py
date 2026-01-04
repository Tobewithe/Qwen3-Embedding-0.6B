from transformers import (
    Trainer,
    TrainingArguments
)
import transformers
from sentence_transformers import SentenceTransformer
import torch
from ClassificationModel import ClassificationModel, ClassificationConfig
print("Transformers version:", transformers.__version__)
print("Transformers path:", transformers.__file__)
def main():
    # 模型
    st_model = SentenceTransformer("output_qwen3_st_finetuned")
    backbone = st_model._first_module().auto_model  # 这就是微调过的 BERT-like 模型
    tokenizer = st_model.tokenizer
    tokens = tokenizer("Hello world")
    print(tokens["input_ids"]) # 分词后添加了<|endoftext|> [9707, 1879, 151643]
    print("Padding side:", tokenizer.padding_side) # Padding side: right
    print("pad token:", tokenizer.pad_token) # pad token: <|endoftext|>
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cls_config = ClassificationConfig(
        num_labels=2,
        backbone_config_dict=backbone.config.to_dict()
    )
    model_for_cls = ClassificationModel(cls_config, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_for_cls.to(device)
    import pandas as pd
    from datasets import Dataset
    # 1. 读取 CSV
    df = pd.read_csv("community_cleaned.csv")  # 替换为你的文件路径
    INSTRUCTION = "Identify if this comment violates the community guidelines: "
    def build_input(row):
        query_part = f"[{row['subreddit']}] {row['rule']}"
        return f"Instruct: {INSTRUCTION}\nQuery: {query_part}\nComment: {row['cleaned_sample']}"
    df["text"] = df.apply(build_input, axis=1)
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)
    # 4. 转为 Hugging Face Dataset
    cls_dataset = Dataset.from_pandas(df)
    # 验证
    print(cls_dataset)
    print(cls_dataset[0])  # 应输出: {'text': '...', 'label': 1}
    def tokenize(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        enc["labels"] = examples["label"]
        return enc
    tokenized_ds = cls_dataset.map(tokenize, batched=True) # (input_ids, attention_mask, labels)

    # 4. 使用 Hugging Face Trainer 训练
    training_args = TrainingArguments(
        output_dir="./cls_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model_for_cls,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        # 可添加 eval_dataset, compute_metrics 等
    )
    trainer.train()
    # 保存模型权重
    model_for_cls.save_pretrained("./cls_output")

    # 保存 tokenizer
    tokenizer.save_pretrained("./cls_output")
