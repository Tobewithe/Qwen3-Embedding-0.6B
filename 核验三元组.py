import pandas as pd
from typing import Dict

_id_cache: Dict[str, Dict[int, Dict]] = {}

def _get_id_map(csv_path: str) -> Dict[int, Dict]:
    id_map = _id_cache.get(csv_path)
    if id_map is None:
        df = pd.read_csv(csv_path)
        required = {'cleaned_sample', 'ID', 'subreddit', 'rule', 'label'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"CSV 缺少列: {missing}")
        id_map = {}
        for _, row in df.iterrows():
            rid = int(row['ID'])
            id_map[rid] = {
                'cleaned_sample': str(row['cleaned_sample']),
                'subreddit': str(row['subreddit']).strip(),
                'rule': str(row['rule']).strip(),
                'label': int(row['label']),
            }
        _id_cache[csv_path] = id_map
    return id_map

def verify_ex(ex, csv_path: str) -> bool:
    id_map = _get_id_map(csv_path)

    texts = getattr(ex, 'texts', None)
    meta = getattr(ex, 'meta', None)
    if not texts:
        print("ex.texts 缺失")
        return False
    if len(texts) != 3:
        print(f"ex.texts 长度为 {len(texts)}，期望 3")
        return False
    if not meta:
        print("ex.meta 缺失")
        return False
    missing = [k for k in ('pos_id', 'neg_id') if k not in meta]
    if missing:
        print(f"ex.meta 缺少字段 {missing}")
        return False

    _, positive, negative = texts

    pos_row = id_map.get(int(meta['pos_id']))
    neg_row = id_map.get(int(meta['neg_id']))
    valid = True
    if pos_row is None:
        print(f"POS id 未在 CSV 中找到: id={meta['pos_id']}")
        valid = False
    if neg_row is None:
        print(f"NEG id 未在 CSV 中找到: id={meta['neg_id']}")
        valid = False
    if pos_row is not None:
        if positive != pos_row['cleaned_sample']:
            print(f"POS 文本不匹配: id={meta['pos_id']}, ex={repr(positive)}, csv={repr(pos_row['cleaned_sample'])}")
            valid = False
        if 'subreddit' in meta and str(meta['subreddit']) != pos_row['subreddit']:
            print(f"POS subreddit 不匹配: id={meta['pos_id']}, meta={meta['subreddit']}, csv={pos_row['subreddit']}")
            valid = False
        if 'rule' in meta and str(meta['rule']) != pos_row['rule']:
            print(f"POS rule 不匹配: id={meta['pos_id']}, meta={meta['rule']}, csv={pos_row['rule']}")
            valid = False
        if pos_row['label'] != 1:
            print(f"POS label 不为 1: id={meta['pos_id']}, csv={pos_row['label']}")
            valid = False
    if neg_row is not None:
        if negative != neg_row['cleaned_sample']:
            print(f"NEG 文本不匹配: id={meta['neg_id']}, ex={repr(negative)}, csv={repr(neg_row['cleaned_sample'])}")
            valid = False
        if 'subreddit' in meta and str(meta['subreddit']) != neg_row['subreddit']:
            print(f"NEG subreddit 不匹配: id={meta['neg_id']}, meta={meta['subreddit']}, csv={neg_row['subreddit']}")
            valid = False
        if 'rule' in meta and str(meta['rule']) != neg_row['rule']:
            print(f"NEG rule 不匹配: id={meta['neg_id']}, meta={meta['rule']}, csv={neg_row['rule']}")
            valid = False
        if neg_row['label'] != 0:
            print(f"NEG label 不为 0: id={meta['neg_id']}, csv={neg_row['label']}")
            valid = False
    if pos_row is not None and neg_row is not None:
        if pos_row['subreddit'] != neg_row['subreddit']:
            print(f"POS/NEG subreddit 不一致: pos={pos_row['subreddit']}, neg={neg_row['subreddit']}")
            valid = False
        if pos_row['rule'] != neg_row['rule']:
            print(f"POS/NEG rule 不一致: pos={pos_row['rule']}, neg={neg_row['rule']}")
            valid = False
    return valid
