import pandas as pd

df = pd.read_csv('community_cleaned.csv')
# 将空字符串 '' 显式转换为缺失值（pd.NA）
df = df.replace('', pd.NA)

print("各列缺失值数量：")
print(df.isnull().sum())

print("\n是否存在缺失值？", df.isnull().any().any())



