import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        enc_output, hidden = self.encoder(src)
        output = self.decoder(trg, hidden, enc_output, teacher_forcing_ratio)
        return output


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, hidden, enc_output, teacher_forcing_ratio=0.5):
        embedded = self.embedding(trg)
        output, (hidden, cell) = self.rnn(embedded, hidden)
        output = self.fc_out(output)
        return output
