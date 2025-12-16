from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros([k,d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(seq_len)], requires_grad=False).unsqueeze(1).repeat(1, k)
        s = 0.0
        interval = seq_len / k
        mu = []
        for _ in range(k):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(k)]).unsqueeze(0))

    def normal_pdf(self, pos, mu, sigma):
        mu = mu.to(device)
        sigma = sigma.to(device)
        pos = pos.to(device)
        a = pos - mu
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(pdfs, self.embedding)
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout,seq_len):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self._attention = nn.MultiheadAttention(seq_len, _heads, batch_first=True)

        self.attn_norm = nn.LayerNorm(d_model)

        self.cnn_units = seq_len

        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1), stride=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src = self.attn_norm(src + self.attention(src, src, src)[0] + self._attention(src.transpose(-1, -2), src.transpose(-1, -2), src.transpose(-1, -2))[0].transpose(-1, -2))
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layer=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src

class ClassificationKeystrokeModel(nn.Module):
    def __init__(self, num_layer, d_model, k, heads, _heads, seq_len, num_classes, inner_dropout):
        super(ClassificationKeystrokeModel, self).__init__()

        self.pos_encoding = PositionalEncoding(k, d_model, seq_len)
        self.encoder = TransformerEncoder(d_model, heads, _heads, seq_len, num_layer, inner_dropout)

        self.ff = nn.Sequential(
            nn.Linear(seq_len * d_model, 512),
            nn.ReLU(),
            nn.Dropout(inner_dropout),
            nn.Linear(512, num_classes)  # num_classes = nombre d'émotions ou classes
        )


    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)
        enc_out = self.encoder(encoded_inputs)
        flattened = torch.flatten(enc_out, start_dim=1, end_dim=2)  # Flatten to [batch_size, seq_len * d_model]
        logits = self.ff(flattened)  # [batch_size, num_classes]
        return logits