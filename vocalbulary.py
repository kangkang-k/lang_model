from collections import Counter

from wash import train_conversations


def build_vocab(conversations):
    word_count = Counter()

    # 统计所有对话中的词频
    for input_tokens, target_tokens in conversations:
        word_count.update(input_tokens)
        word_count.update(target_tokens)

    # 给词汇表中的每个单词分配一个索引
    vocab = {word: idx for idx, (word, _) in enumerate(word_count.items(), start=4)}  # 索引从4开始，0-3留给特殊符号
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3  # 未知词

    return vocab


# 构建词汇表
vocab = build_vocab(train_conversations)
print(f"Vocabulary size: {len(vocab)}")
