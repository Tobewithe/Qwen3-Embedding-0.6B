import pandas as pd

# 读取原始 CSV 文件
df = pd.read_csv('test.csv')
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
# 保存到新 CSV 文件
output_df.to_csv('processed_test.csv', index=False)

