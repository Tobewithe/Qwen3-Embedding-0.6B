import re
import emoji
import pandas as pd

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

df = pd.read_csv('processed_train.csv')
df['cleaned_sample'] = df['sample'].apply(clean_for_violation_detection)
before = len(df)
df = df.drop_duplicates(subset=['subreddit','rule', 'cleaned_sample'], keep='first')
after = len(df)
df = df.reset_index(drop=True)
df['ID'] = range(1, len(df) + 1)
df.to_csv('community_cleaned.csv', index=False)
print(f"去重前: {before}, 去重后: {after}, 删除: {before - after} 行")
print("社区发言清洗完成！")
