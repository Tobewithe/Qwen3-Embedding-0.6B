# inference_cls.py
import torch
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Union
import os
from ClassificationModel import ClassificationModel
import pandas as pd
INSTRUCTION = "Identify if this comment violates the community guidelines: "

def load_full_model(model_dir: str, device: str = "cuda"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    model = ClassificationModel.from_pretrained(
        model_dir,
        trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# ----------------------------
# 3. 推理函数（支持单条/多条文本）
# ----------------------------
def predict(
    model,
    tokenizer,
    texts: Union[str, List[str]],
    device: str = "cpu",
    max_length: int = 512
):
    was_single = isinstance(texts, str)
    if was_single:
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
        return_attention_mask=True
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs)
        logits = out["logits"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

    if was_single:
        return preds[0].item(), probs[0][preds[0]].item()
    else:
        preds_list = preds.detach().cpu().tolist()
        confs_tensor = probs[torch.arange(len(preds)), preds]
        confs_list = confs_tensor.detach().cpu().tolist()
        return preds_list, confs_list

# ----------------------------
# 3.1 组装输入（参考 HF 训练）
# ----------------------------
def build_texts_from_csv(
    csv_path: str,
    limit: int = None
) -> List[str]:
    df = pd.read_csv(csv_path)
    required = {"subreddit", "rule", "cleaned_sample"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV 缺少列: {missing}")
    if limit is not None:
        df = df.head(limit)
    texts = []
    row_ids = []
    for _, row in df.iterrows():
        subred = str(row["subreddit"]).strip()
        rule = str(row["rule"]).strip()
        comment = str(row["cleaned_sample"])
        row_id = int(row["row_id"])
        
        if not subred or not rule or not comment:
            continue
        query = f"[{subred}] {rule}"
        text = f"Instruct: {INSTRUCTION}\nQuery: {query}\nComment: {comment}"
        row_ids.append(row_id)
        texts.append(text)
    return texts, row_ids
def classify_from_csv(
    model_dir: str,
    csv_path: str,
    device: str = "cuda",
    max_length: int = 512,
    fp16: bool = False,
    limit: int = None
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, tokenizer = load_full_model(model_dir, device)
    if fp16 and device == "cuda":
        model.half()
    texts, row_ids = build_texts_from_csv(csv_path, limit)
    preds, confs = predict(model, tokenizer, texts, device, max_length=max_length)
    return preds, confs, texts, row_ids

def main():
    preds, confs, texts, row_ids = classify_from_csv("./cls_output", "test_cleaned.csv", device="cuda", max_length=512, fp16=False, limit=10)
    for i, text in enumerate(texts):
        pred = preds[i] if isinstance(preds, list) else preds
        conf = confs[i] if isinstance(confs, list) else confs
        print(f"文本: {text}")
        print(f"预测类别: {pred} | 置信度: {conf:.4f}\n")
        print(f"row_id: {row_ids[i]}\n")
if __name__ == "__main__":
    main()
    
