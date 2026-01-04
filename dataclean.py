
def split_data(csv):
    # 读取原始 CSV 文件
    df = pd.read_csv(csv)
    # 初始化空列表用于存储新行
    new_rows = []
    # 遍历每一行
    for _, row in df.iterrows():
        row_id = row['row_id']
        rule = row['rule']
        subreddit = row['subreddit']
        body = row['body']
        rule_violation = row.get('rule_violation')  # body 的标签（测试集可能无该列）

        # 添加 body 作为样本（标签为 rule_violation）
        new_rows.append({
            'sample': body,
            'label': rule_violation,
            'row_id': row_id,
            'rule': rule,
            'subreddit': subreddit
        })

        # 添加 positive_example_1 和 positive_example_2（标签为 1）
        for col in ['positive_example_1', 'positive_example_2']:
            example = row[col]
            if pd.notna(example):  # 忽略 NaN 值
                new_rows.append({
                    'sample': example,
                    'label': 1,
                    'row_id': row_id,
                    'rule': rule,
                    'subreddit': subreddit
                })

        # 添加 negative_example_1 和 negative_example_2（标签为 0）
        for col in ['negative_example_1', 'negative_example_2']:
            example = row[col]
            if pd.notna(example):  # 忽略 NaN 值
                new_rows.append({
                    'sample': example,
                    'label': 0,
                    'row_id': row_id,
                    'rule': rule,
                    'subreddit': subreddit
                })
    # 转换为 DataFrame
    output_df = pd.DataFrame(new_rows, columns=['subreddit','rule','sample','label','row_id'])
    output_df['ID'] = range(1, len(output_df) + 1)
    return output_df



def clean_for_violation_detection(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+|http\S+', '<URL>', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text

def main():
    import pandas as pd
    import re
    import emoji
    # 分离数据
    df = split_data('train.csv')
    df['cleaned_sample'] = df['sample'].apply(clean_for_violation_detection)
    before = len(df)
    df = df.drop_duplicates(subset=['subreddit','rule', 'cleaned_sample'], keep='first')
    after = len(df)
    df = df.reset_index(drop=True)
    df['ID'] = range(1, len(df) + 1)
    df.to_csv('community_cleaned.csv', index=False)
    print(f"去重前: {before}, 去重后: {after}, 删除: {before - after} 行")
    print("社区发言清洗完成！")

if __name__ == "__main__":
    main()
