import torch

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from .ConditionalVAE import ConditionalVAE
from utils import normalise


def quick():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max = 512, 64, 64, 32, 2, 1, 64, 0.0015, 100, 0.005
    culture_embed_dim = 32

    model_name = f'ConditionalVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_ced{culture_embed_dim}.pt'

    print(f"Model name: {model_name}")

    checkpoint = torch.load(model_name, map_location=device)

    language_to_id = checkpoint["language_to_id"]
    num_cultures = len(language_to_id)

    vocab = CharVocab(ALLOWED_CHARS)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        name = "Alessandro"
        source_culture = "Italian"

        encoded = vocab.encode(normalise(name))

        x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        length = torch.tensor([x.size(1)], dtype=torch.long, device=device)

        source_label = torch.tensor([language_to_id[source_culture]], dtype=torch.long, device=device)

        z, mu, logvar = model.encoder(x, length, source_label)
        z = mu

        reconstruction = model.decoder.generate(z, source_label)

        print(f"Reconstruction ({source_culture}): {reconstruction}")

        for target_culture in ["Chinese", "Italian", "French", "Japanese", "German"]:
            target_label = torch.tensor([language_to_id[target_culture]], dtype=torch.long, device=device)

            generated = model.decoder.generate(z, target_label)

            print(f"{source_culture} -> {target_culture}: {generated}")


if __name__ == "__main__":
    quick()