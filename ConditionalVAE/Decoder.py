import torch
import torch.nn as nn

from AE.CharVocab import CharVocab
from nGram.nGram import nGram


class Decoder(nn.Module):

    def __init__(self, vocab: CharVocab, embed_dim: int, hidden_dim: int, num_layers: int, latent_dim: int,
                 num_cultures: int, culture_embed_dim: int, culture_embedding: nn.Embedding):
        super().__init__()

        # Character vocabulary
        self.vocab = vocab

        # Dimensionality of character embeddings
        self.embed_dim = embed_dim

        # Dimensionality of the hidden state
        self.hidden_dim = hidden_dim

        # Number of layers
        self.num_layers = num_layers

        # Size of the latent representation
        self.latent_dim = latent_dim

        # Embedding layer with size (len(vocab), embed_dim)
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        # Unidirectional LSTM
        # class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,
        # batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
        # batch_first returns (batch_size, seq_len, hidden_dim)
        self.rnn = nn.LSTM(embed_dim + latent_dim + culture_embed_dim, hidden_dim, num_layers, bias=True,
                           batch_first=True,
                           bidirectional=False)

        # Linear projection from (batch_size, seq_len, hidden_dim) to (batch_size, seq_len, len(vocab))
        self.fc = nn.Linear(self.hidden_dim, len(vocab))

        # Embedding layer with size (num_cultures, culture_embedding_dim)
        self.culture_embedding = culture_embedding

        # Linear projections from (batch_size, latent_dim + culture_embed_dim) to (batch_size, num_layers * hidden_dim)
        self.hidden_init = nn.Linear(latent_dim + culture_embed_dim, num_layers * hidden_dim)
        self.cell_init = nn.Linear(latent_dim + culture_embed_dim, num_layers * hidden_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        culture_embedding = self.culture_embedding(labels)

        # Culture dropout
        '''
        if self.training:
            mask = torch.rand(culture_embedding.size(0), device=culture_embedding.device) < 0.15
            culture_embedding[mask] = 0
        '''

        decoder_condition = torch.cat([z, culture_embedding], dim=-1)

        # z is (batch_size, latent_dim)
        # h0 and c0 are (batch_size, num_layers * hidden_dim)
        h0 = self.hidden_init(decoder_condition)
        c0 = self.cell_init(decoder_condition)

        batch_size = z.size(0)

        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = c0.view(self.num_layers, batch_size, self.hidden_dim)

        # Character dropout
        if self.training:
            mask = torch.rand(x.shape, device=x.device) < 0.25
            x = x.clone()
            x[mask] = self.vocab.char2idx['<MASK>']

        # At every timestep, the RNN takes both x and z
        emb = self.embedding(x)
        z_rep = z.unsqueeze(1).repeat(1, emb.size(1), 1)
        culture_rep = culture_embedding.unsqueeze(1).repeat(1, emb.size(1), 1)
        rnn_input = torch.cat([emb, z_rep, culture_rep], dim=-1)

        # out is (batch_size, seq_len, hidden_dim)
        out, (_, _) = self.rnn(rnn_input, (h0, c0))

        # Logits are (batch_size, seq_len, len(vocab))
        logits = self.fc(out)

        return logits, out

    @torch.no_grad()
    def generate(
            self,
            z: torch.Tensor,
            culture_embedding: torch.Tensor,
            max_len=50,
            beam_size=1,
            ngram2=None,
            ngram3=None,
            ngram4=None,
            cultures=None,
            culture_weights=None
    ):
        decoder_condition = torch.cat([z, culture_embedding], dim=-1)

        batch_size = z.size(0)

        h0 = self.hidden_init(decoder_condition)
        c0 = self.cell_init(decoder_condition)

        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = c0.view(self.num_layers, batch_size, self.hidden_dim)

        eos_token = self.vocab.char2idx['<EOS>']
        sos_token = self.vocab.char2idx['<SOS>']

        ngram_vocabulary = [
            '>' if self.vocab.idx2char[i] == '<EOS>'
            else self.vocab.idx2char[i]
            for i in range(len(self.vocab))
        ]

        beta2, beta3, beta4 = 0.10, 0.10, 0.10

        results = []

        for i in range(batch_size):

            # Each beam:
            # (tokens, score, h, c, finished)
            beams = [
                (
                    [],
                    0.0,
                    h0[:, i:i + 1, :],
                    c0[:, i:i + 1, :],
                    False
                )
            ]

            for _ in range(max_len):

                candidates = []

                for tokens, score, h, c, finished in beams:

                    # Don't expand beams that already produced EOS
                    if finished:
                        candidates.append(
                            (tokens, score, h, c, True)
                        )
                        continue

                    # Previous token
                    if len(tokens) == 0:
                        x = torch.tensor(
                            [[sos_token]],
                            dtype=torch.long,
                            device=z.device
                        )
                    else:
                        x = torch.tensor(
                            [[tokens[-1]]],
                            dtype=torch.long,
                            device=z.device
                        )

                    emb = self.embedding(x)

                    z_i = z[i:i + 1]
                    culture_i = culture_embedding[i:i + 1]

                    rnn_input = torch.cat(
                        [emb, z_i.unsqueeze(1), culture_i.unsqueeze(1)],
                        dim=-1
                    )

                    out, (new_h, new_c) = self.rnn(
                        rnn_input,
                        (h, c)
                    )

                    logits = self.fc(out[:, -1])

                    # Convert CVAE logits into log probabilities
                    cvae_log_probs = torch.log_softmax(
                        logits,
                        dim=-1
                    ).squeeze(0)

                    # -------------------------------------------------
                    # N-gram guidance
                    # -------------------------------------------------

                    if (
                        ngram2 is not None
                        and ngram3 is not None
                        and ngram4 is not None
                        and cultures is not None
                    ):

                        prefix = '<' + self.vocab.decode(tokens)

                        guided_log_probs = cvae_log_probs.clone()

                        if len(prefix) >= 1:
                            log_probs_2 = torch.tensor(
                                ngram2.next_char_log_probabilities(
                                    prefix,
                                    cultures,
                                    ngram_vocabulary,
                                    culture_weights
                                ),
                                dtype=torch.float32,
                                device=z.device
                            )

                            guided_log_probs += beta2 * log_probs_2

                        if len(prefix) >= 2:
                            log_probs_3 = torch.tensor(
                                ngram3.next_char_log_probabilities(
                                    prefix,
                                    cultures,
                                    ngram_vocabulary,
                                    culture_weights
                                ),
                                dtype=torch.float32,
                                device=z.device
                            )

                            guided_log_probs += beta3 * log_probs_3

                        if len(prefix) >= 3:
                            log_probs_4 = torch.tensor(
                                ngram4.next_char_log_probabilities(
                                    prefix,
                                    cultures,
                                    ngram_vocabulary,
                                    culture_weights
                                ),
                                dtype=torch.float32,
                                device=z.device
                            )

                            guided_log_probs += beta4 * log_probs_4

                    else:
                        guided_log_probs = cvae_log_probs

                    # -------------------------------------------------
                    # Keep best next tokens
                    # -------------------------------------------------

                    top_log_probs, top_tokens = torch.topk(
                        guided_log_probs,
                        beam_size
                    )

                    for j in range(beam_size):

                        token = top_tokens[j].item()
                        token_log_prob = top_log_probs[j].item()

                        new_tokens = tokens.copy()

                        new_finished = (
                            token == eos_token
                        )

                        if not new_finished:
                            new_tokens.append(token)

                        new_score = score + token_log_prob

                        candidates.append(
                            (
                                new_tokens,
                                new_score,
                                new_h.clone(),
                                new_c.clone(),
                                new_finished
                            )
                        )

                # -----------------------------------------------------
                # Keep the best beams
                # -----------------------------------------------------

                candidates.sort(
                    key=lambda beam: beam[1] / (max(len(beam[0]), 1)),
                    reverse=True
                )

                beams = candidates[:beam_size]

                # Stop if every beam has finished
                if all(beam[4] for beam in beams):
                    break

            # ---------------------------------------------------------
            # Length-normalised final ranking
            # ---------------------------------------------------------

            best_beam = max(
                beams,
                key=lambda beam: (
                    beam[1] / max(len(beam[0]), 1)
                )
            )

            results.append(
                self.vocab.decode(best_beam[0])
            )

        return results