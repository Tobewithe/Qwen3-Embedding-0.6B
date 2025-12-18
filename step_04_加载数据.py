import pandas as pd
import random
from collections import defaultdict
from torch.utils.data import Dataset
from sentence_transformers import InputExample
# 查询指令前缀（可根据任务场景自定义）
INSTRUCTION = "Identify if this comment violates the community guidelines: "
class ContrastiveCommunityDataset(Dataset):
    def __init__(self, csv_path, instruction_prefix=INSTRUCTION, shuffle_pairs=True):
        self.instruction_prefix = instruction_prefix
        self.triplets = []

        df = pd.read_csv(csv_path)
        assert "subreddit" in df.columns, "缺少 subreddit 字段"
        assert "rule" in df.columns, "缺少 rule 字段"
        assert "label" in df.columns, "缺少 label 字段"
        assert "cleaned_sample" in df.columns, "缺少 cleaned_sample 字段"
        assert "ID" in df.columns, "缺少 ID 字段"
        # groups 结构：按 subreddit → rule 分桶，桶内仅保存样本的 ID 集合
        groups = defaultdict(lambda: defaultdict(lambda: {"pos_ids": set(), "neg_ids": set()}))
        # id_to_text：ID → cleaned_sample 的映射，用于 __getitem__ 时回查文本
        self.id_to_text = {}
        for _, row in df.iterrows():
            subred = row.get("subreddit")
            rule_text = row.get("rule")
            label = row.get("label")
            text = row.get("cleaned_sample")
            rid = row.get("ID")
            if pd.isna(subred) or pd.isna(text) or pd.isna(rule_text):
                continue
            subred = str(subred).strip()
            rule_text = str(rule_text).strip()
            text = str(text).strip()
            if not subred or not text or not rule_text:
                continue
            self.id_to_text[int(rid)] = text
            if label == 1:
                groups[subred][rule_text]["pos_ids"].add(int(rid))
            elif label == 0:
                groups[subred][rule_text]["neg_ids"].add(int(rid))
        for subred, rules in groups.items():
            for rule_text, data in rules.items():
                pos_ids = list(data["pos_ids"])
                neg_ids = list(data["neg_ids"])
                if not pos_ids or not neg_ids:
                    continue
                # 可选：打乱 ID 顺序，提升配对的多样性
                if shuffle_pairs:
                    random.shuffle(pos_ids)
                    random.shuffle(neg_ids)
                # 只取两边最小数量，逐一配对 ID
                n = min(len(pos_ids), len(neg_ids))
                for i in range(n):
                    p_id = pos_ids[i]
                    n_id = neg_ids[i]
                    self.triplets.append((subred, rule_text, p_id, n_id))
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # 通过 ID 回查文本，构造真实训练样本，同时保留元信息
        subred, rule_text, pos_id, neg_id = self.triplets[idx]
        anchor = f"Instruct: {self.instruction_prefix}\nQuery: [{subred}] {rule_text}"
        positive = self.id_to_text[int(pos_id)]
        negative = self.id_to_text[int(neg_id)]
        ex = InputExample(texts=[anchor, positive, negative])
        ex.meta = {"pos_id": int(pos_id), "neg_id": int(neg_id), "subreddit": str(subred), "rule": str(rule_text)}
        return ex
