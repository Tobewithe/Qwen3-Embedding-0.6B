import pandas as pd
from torch._refs import to
df = pd.read_csv('processed_train.csv')
# 获取除 'ID' 外的所有列
cols_to_check = ['subreddit','rule','sample','label']
# 查看是否有任何重复行
has_duplicates = df.duplicated(subset=cols_to_check).any()
print("是否存在重复行:", has_duplicates)

