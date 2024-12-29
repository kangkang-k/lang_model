import torch
import torch.optim as optim
from torch import nn

from seq2seq import Seq2Seq
from swap import train_tensor
from vocalbulary import vocab


# 训练模型函数
def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for input_tensor, target_tensor in train_data:
        optimizer.zero_grad()

        # 输入数据和目标数据
        src = input_tensor.unsqueeze(0)  # 添加批次维度
        trg = target_tensor.unsqueeze(0)  # 添加批次维度

        # 前向传播
        output = model(src, trg)

        # 计算损失
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_data)


# 超参数
input_dim = len(vocab)
output_dim = len(vocab)
emb_dim = 256
hidden_dim = 512
n_layers = 3
dropout = 0.1

# 确保设备正确设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

# 训练
num_epochs = 10
clip = 1

for epoch in range(num_epochs):
    train_loss = train(model, train_tensor, optimizer, criterion, clip)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
