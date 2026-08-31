import json

import torch
import os

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE


def export():
    device = torch.device("cpu")

    # Model hyperparameters
    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    culture_embed_dim = 64

    vocab = CharVocab(ALLOWED_CHARS)

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    num_cultures = len(language_to_id)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                       num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)

    model_name = f'CCVAE/best_model_conditional_supcon_logits_bi_bs512_ed64_hde64_hdd32_nle2_nld1_ld32_lr0.0005adam_ep100es10_cd0.25_blf0t0.025o5_ced64_t0.1_l0.75.pt'

    model.load_state_dict(torch.load(model_name, map_location=device))

    model.eval()

    batch_size = 2
    seq_len = 20

    dummy_x = torch.randint(0, len(vocab), (batch_size, seq_len), dtype=torch.long)
    dummy_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    dummy_labels = torch.randint(0, num_cultures, (batch_size,), dtype=torch.long)

    try:
        torch.onnx.export(
            model,
            (dummy_x, dummy_lengths, dummy_labels),
            "cvae.onnx",
            input_names=["x", "lengths", "labels"],
            output_names=["logits", "decoder_hidden", "mu", "logvar"],
            dynamic_axes={
                "x": {0: "batch_size", 1: "seq_len"},
                "lengths": {0: "batch_size"},
                "labels": {0: "batch_size"},
            },
            dynamo=False,
        )
        print("Export succeeded:", os.path.abspath("cvae.onnx"))
    except Exception as e:
        print("Export failed:", e)

    print(os.path.abspath("cvae.onnx"))

if __name__ == "__main__":
    export()