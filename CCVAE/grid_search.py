import argparse

from CCVAE.evaluate import evaluate
from CCVAE.train import train


def grid_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    lr = args.lr

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = CharVocab(ALLOWED_CHARS)

    # Model hyperparameters
    batch_size = 512
    embed_dim = 64
    hidden_dim_encoder = 64
    hidden_dim_decoder = 32
    num_layers_encoder = 2
    num_layers_decoder = 1
    latent_dim = 32
    epochs = 50
    patience = 10
    beta_max = 0.025
    n_epochs_ramp_up = 5
    temperature = 0.1
    lambda_supcon = 0.75
    culture_embed_dim = 64

    with open("language_to_id.json", "r") as f:
        language_to_id = json.load(f)

    model_name = (f'CCVAE/models/best_model_conditional_supcon_logits_'
                  f'bi_'
                  f'bs{batch_size}_'
                  f'ed{embed_dim}_'
                  f'hde{hidden_dim_encoder}_'
                  f'hdd{hidden_dim_decoder}_'
                  f'nle{num_layers_encoder}_'
                  f'nld{num_layers_decoder}_'
                  f'ld{latent_dim}_'
                  f'lr{lr}adam_'
                  f'ep{epochs}es{patience}_'
                  f'cd0.25_'
                  f'blf0t{beta_max}o{n_epochs_ramp_up}_'
                  f'ced{culture_embed_dim}_'
                  f't{temperature}_'
                  f'l{lambda_supcon}.pt'
                  )

    num_cultures = len(language_to_id)

    model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                           num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
    model.load_state_dict(torch.load(model_name, map_location=device))
    '''

    model, train_names = train(lr)

    evaluate(model, lr)


if __name__ == "__main__":
    grid_search()
