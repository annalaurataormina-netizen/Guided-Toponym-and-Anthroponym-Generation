import torch
import torch.nn as nn

from CharVocab import CharVocab


class Decoder(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        # Character vocabulary
        self.vocab = vocab

        # Dimensionality of character embeddings
        self.embed_dim = embed_dim

        # Dimensionality of the hidden state
        self.hidden_dim = hidden_dim

        # Number of layers
        self.num_layers = num_layers

        # Embedding layer with size (len(vocab), embed_dim)
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        # Unidirectional LSTM
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        # batch_first returns (batch_size, seq_len, hidden_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, bias=True, batch_first=True, bidirectional=False)

        # Linear projection that returns (batch_size, seq_len, len(vocab))
        self.fc = nn.Linear(self.hidden_dim, len(vocab))

    def forward(self, x: torch.Tensor, hn_encoder: torch.Tensor, cn_encoder: torch.Tensor) -> torch.Tensor:
        # Initial hidden state and cell state
        h0, c0 = hn_encoder, cn_encoder

        # out is (batch_size, seq_len, hidden_dim)
        # hn, cn are (num_layers, batch_size, hidden_dim)
        out, (hn, cn) = self.rnn(self.embedding(x), (h0, c0))

        # Logits are (batch_size, seq_len, len(vocab))
        return self.fc(out)

    @torch.no_grad()
    def generate(self, hn_encoder: torch.Tensor, cn_encoder: torch.Tensor, max_len=50) -> str:

        # Batch size inferred from hidden state
        batch_size = hn_encoder.size(1)

        # Start with <SOS>
        x = torch.full((batch_size, 1), self.vocab.char2idx['<SOS>'],
                       dtype=torch.long,
                       device=hn_encoder.device)

        h, c = hn_encoder, cn_encoder
        generated = []

        for _ in range(max_len):

            # One decoding step
            embedded = self.embedding(x)
            out, (h, c) = self.rnn(embedded, (h, c))
            logits = self.fc(out[:, -1])  # (batch, vocab_size)

            # Greedy decoding
            x = logits.argmax(dim=-1, keepdim=True)

            token = x.item()

            generated.append(token)

            if token == self.vocab.char2idx['<EOS>']:
                break

        return self.vocab.decode(generated)
