数据分离脚本：输入：train.csv  输出：processed_train.csv 作用把train里的对比数据分离出来变成单个样本
数据清洗： 输入:processed_train.csv 输出：community_cleaned.csv 作用数据清洗，增加一列cleaned_sample
ST训练:main("community_cleaned.csv","output_qwen3_st_finetuned"),第一个参数是训练集，第二个参数是模型输出。 作用训练Qwen3-Embedding-0.6B模型， model = SentenceTransformer("./model")这里要改为你自己下载的Qwen3-Embedding-0.6B模型路径
HF训练：输入：st_model = SentenceTransformer("output_qwen3_st_finetuned")，输出："./cls_output"已经硬编码成这个路径。
推理：classify_from_csv("./cls_output", "test_cleaned.csv", device="cuda", max_length=512, fp16=False, limit=10)，把清理后的测试集路径填上就行。

