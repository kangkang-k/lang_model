import nltk
from nltk.tokenize import word_tokenize
import string
import os
import nltk
nltk.download('punkt_tab')

nltk.download('punkt')


# 预处理函数：去掉标点、转换为小写、分词
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens


# 读取并清理数据
def load_data(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(0, len(lines) - 1, 2):  # 每两个连续的句子是一个对话
        input_text = lines[i].strip()
        target_text = lines[i + 1].strip()

        # 分词并存储
        input_tokens = preprocess_text(input_text)
        target_tokens = preprocess_text(target_text)

        conversations.append((input_tokens, target_tokens))

    return conversations


# 读取训练数据
train_file = r'D:\PyCharmProject\xibaluoma\ijcnlp_dailydialog\dialogues_text.txt'
train_conversations = load_data(train_file)
