import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ContrastiveVAE.NameDataset import NameDataset
from ContrastiveVAE.losses import SupConLoss

# Temperature

def test(model, test_names, vocab):

    # Test dataset
    test_dataset = NameDataset(test_names, vocab)

    test_dataloader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False)

    # Use cross entropy loss to train the model, ignoring <PAD> characters
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

    # SupCon criterion
    supcon_criterion = SupConLoss(temperature=temperature)

    # Tracks the number of batches
    global_step = 0

    # Loss tracking
    total_reconstruction_loss = 0
    total_kl_loss = 0
    total_supcon_loss = 0

    # Levenshtein distance tracking
    total_lev = 0
    total_count = 0

    device = model.device

    with torch.no_grad():
        for test_batch in test_dataloader:

            # Batch Levenshtein distance
            batch_lev = 0
            batch_count = 0

            sequences, lengths, labels = test_batch
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)

            # Drop <SOS> as it can only ever be a starting input, never a valid target to predict
            # target is (batch, seq_len)
            target = sequences[:, 1:]

            # Forward pass
            # logits, mu, logvar are (batch_size, seq_len, len(vocab))
            logits, _, mu, logvar = model(sequences, lengths, labels)

            # Convert predicted indices to characters
            pred_indices = logits.argmax(dim=-1)

            for p, t in zip(pred_indices, target):
                eos_idx = vocab.char2idx['<EOS>']

                # Remove everything after <EOS>
                p_list = p.tolist()
                p_list = p_list[:p_list.index(eos_idx)] if eos_idx in p_list else p_list
                pred_str = vocab.decode(p_list)
                target_str = vocab.decode(t.tolist())

                # Normalised Levenshtein distance
                distance = editdistance.eval(pred_str, target_str) / max(len(pred_str), len(target_str))

                batch_lev += distance
                total_lev += distance
                batch_count += 1
                total_count += 1

            # reshape converts logits from (batch, seq_len, len(vocab)) to (batch * seq_len, len(vocab))
            # reshape converts target from (batch, seq_len) to (batch * seq_len,)
            # CrossEntropyLoss internally applies log_softmax to logits and computes the negative log likelihood loss
            reconstruction_loss = criterion(
                logits.reshape(-1, len(vocab)),
                target.reshape(-1)
            )

            # KL divergence
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            # SupCon loss
            pad_idx = vocab.char2idx['<PAD>']
            mask = (target != pad_idx).unsqueeze(-1)  # (batch, seq_len, 1)
            embedding = (logits * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embedding = F.normalize(embedding, dim=1)
            features = embedding.unsqueeze(1)
            supcon_loss = supcon_criterion(features, labels)

            global_step += 1

            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            total_supcon_loss += supcon_loss.item()

            '''
            print(
                f"Step {global_step}, "
                f"Reconstruction loss = {reconstruction_loss.item():.4f}, "
                f"KL divergence = {kl_loss.item():.4f}, "
                f"SupCon loss = {supcon_loss.item():.4f}, "
                f"Avg normalised Levenshtein distance: {batch_lev / batch_count:.4f}"
            )
            '''

    return({"Avg reconstruction loss": total_reconstruction_loss / len(test_dataloader),
            "Avg KL divergence": total_kl_loss / len(test_dataloader),
            "Avg SupCon loss": total_supcon_loss / len(test_dataloader),
            "Avg normalised Levenshtein distance": total_lev / total_count
    })
