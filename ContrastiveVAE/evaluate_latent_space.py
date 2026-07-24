import numpy as np
import torch
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ContrastiveVAE.ContrastiveVAE import ContrastiveVAE
from ContrastiveVAE.NameDataset import NameDataset
from CultureClassifier.LatentExtractor import LatentExtractor
from utils import load_all, normalise

'''
IN ORDER TO RUN, ADJUST THE HYPERPARAMETERS BELOW SO THAT THE RIGHT MODEL IS LOADED.
'''


def evaluate_latent_space():

    seed = 1996

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # VAE hyperparameters
    '''
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 64, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    '''

    # ContrastiveVAE hyperparameters
    batch_size, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder, latent_dim, lr, epochs, beta_max, n_epochs_ramp_up = 512, 32, 64, 32, 2, 1, 64, 0.0015, 30, 0.005, 5
    proj_hidden_dim, proj_output_dim, temperature, lambda_supcon = 128, 64, 0.1, 0.25

    vocab = CharVocab(ALLOWED_CHARS)

    # Load data
    names = load_all(culture=True)

    language_to_id = {
        lang: i
        for i, lang in enumerate(sorted(set(n[1] for n in names)))
    }

    names_normalised = [
        [normalise(name), language_to_id[lang]]
        for name, lang in names
    ]

    dataset = NameDataset(names_normalised, vocab)

    # Load model
    # ContrastiveVAE
    model = ContrastiveVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, proj_hidden_dim, proj_output_dim)
    model_name = f'ContrastiveVAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}_phd{proj_hidden_dim}_pod{proj_output_dim}_t{temperature}_l{lambda_supcon}.pt'

    # VAE
    '''
    model = VAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder, num_layers_decoder,
                latent_dim)
    model_name = f'VAE/models/best_model_bs{batch_size}_ed{embed_dim}_hde{hidden_dim_encoder}_hdd{hidden_dim_decoder}_nle{num_layers_encoder}_nld{num_layers_decoder}_ld{latent_dim}_lr{lr}_ep{epochs}_blf0t{beta_max}.pt'
    '''

    print(model_name)

    model.load_state_dict(torch.load(model_name, map_location=device))

    model.to(device)
    model.eval()

    # Extract latent means
    extractor = LatentExtractor(model.encoder)

    latents, labels = extractor.extract(dataset, batch_size, device)

    # Move to numpy
    X = latents.cpu().numpy()
    y = labels.cpu().numpy()

    print("Total samples:", len(X))
    print("Number of cultures:", len(np.unique(y)))

    # --------------------------------
    # Stratified subsample for geometry metrics
    # --------------------------------

    max_samples = 50000

    if len(X) > max_samples:

        rng = np.random.default_rng(seed)

        sampled_indices = []

        for culture in np.unique(y):
            culture_indices = np.where(y == culture)[0]

            # Keep proportional representation
            n = int(
                max_samples * len(culture_indices) / len(X)
            )

            n = max(n, 2)  # silhouette requires multiple samples per class

            sampled_indices.extend(
                rng.choice(
                    culture_indices,
                    size=min(n, len(culture_indices)),
                    replace=False
                )
            )

        sampled_indices = np.array(sampled_indices)

        X_eval = X[sampled_indices]
        y_eval = y[sampled_indices]

    else:
        X_eval = X
        y_eval = y

    print("Samples used for geometry metrics:", len(X_eval))

    # -----------------------------
    # Overall silhouette
    # -----------------------------

    score = silhouette_score(X_eval, y_eval, metric="euclidean")

    print()
    print("Overall silhouette score:")
    print(score)

    # -----------------------------
    # Per culture silhouette
    # -----------------------------

    sample_scores = silhouette_samples(X_eval, y_eval, metric="euclidean")

    print()
    print("Per culture silhouette:")

    for culture in np.unique(y_eval):
        mask = y_eval == culture

        print(
            f"Culture {culture}: "
            f"{sample_scores[mask].mean():.4f} "
            f"(n={mask.sum()})"
        )

    # -----------------------------
    # kNN purity
    # -----------------------------

    k = 10

    neighbours = NearestNeighbors(
        n_neighbors=k + 1
    )

    neighbours.fit(X_eval)

    _, indices = neighbours.kneighbors(X_eval)

    purity = []

    for i in range(len(X_eval)):
        neighbour_labels = y_eval[indices[i][1:]]

        purity.append(
            np.mean(neighbour_labels == y_eval[i])
        )

    print()
    print(f"{k}-NN purity:")
    print(np.mean(purity))


if __name__ == "__main__":
    evaluate_latent_space()
