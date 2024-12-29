import torch

from vocalbulary import vocab
from wash import train_conversations


def text_to_tensor(tokens, vocab):
    return torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokens], dtype=torch.long)

# 示例：将训练数据转换为张量
train_tensor = [(text_to_tensor(input_tokens, vocab), text_to_tensor(target_tokens, vocab))
                for input_tokens, target_tokens in train_conversations]
