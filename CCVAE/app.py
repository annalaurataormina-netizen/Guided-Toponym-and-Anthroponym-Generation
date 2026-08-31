import json
from collections import Counter
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse

from AE.CharVocab import CharVocab
from AE.config import ALLOWED_CHARS
from ConditionalVAE.ConditionalVAE import ConditionalVAE
from nGram.nGram import nGram
from utils import load_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = FastAPI()

vocab = CharVocab(ALLOWED_CHARS)

embed_dim = 64
hidden_dim_encoder = 64
hidden_dim_decoder = 32
num_layers_encoder = 2
num_layers_decoder = 1
latent_dim = 32
culture_embed_dim = 64

with open("language_to_id.json", "r") as f:
    language_to_id = json.load(f)

dataset = load_all(culture=True)

culture_counts = Counter(
    language
    for _, language in dataset
)

min_culture_names = 10000

language_to_id_filtered = {l: i for l, i in language_to_id.items() if culture_counts[l] >= min_culture_names}

model_name = "CCVAE/best_model_conditional_supcon_logits_bi_bs512_ed64_hde64_hdd32_nle2_nld1_ld32_lr0.0005adam_ep100es10_cd0.25_blf0t0.025o5_ced64_t0.1_l0.75.pt"

num_cultures = len(language_to_id)

model = ConditionalVAE(vocab, embed_dim, hidden_dim_encoder, hidden_dim_decoder, num_layers_encoder,
                       num_layers_decoder, latent_dim, num_cultures, culture_embed_dim)
model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()

ngram2, ngram3, ngram4 = nGram(2), nGram(3), nGram(4)
ngram2.load()
ngram3.load()
ngram4.load()

n, max_length = 10, 50
temperature, beam_size = 0.6, 10


@app.get("/")
def serve_html():
    return FileResponse("CCVAE/index.html")


@app.get("/script.js")
def serve_js():
    return FileResponse("CCVAE/script.js")


@app.get("/style.css")
def serve_css():
    return FileResponse("CCVAE/style.css")


@app.get("/favicon.png")
def serve_favicon():
    return FileResponse("CCVAE/favicon.png")


@app.get("/api/languages")
def get_languages() -> Dict[str, List[str]]:
    languages = list(language_to_id_filtered.keys())
    return {"languages": languages}


@app.get("/api/languages/{language}/names")
def generate(language: str, number: Optional[int] = None, blend_language: Optional[str] = None,
             weight: Optional[float] = None) -> Dict[str, List[str]]:
    language_id = language_to_id_filtered.get(language, None)

    if language_id is None:
        raise HTTPException(status_code=404, detail="Item not found")

    if blend_language is None or (blend_language is not None and language == blend_language):
        if number is not None:
            names = model.generate(culture=language_id, n=number * 5, max_length=max_length, temperature=temperature,
                                   beam_size=beam_size, cultures=[language])
        else:
            names = model.generate(culture=language_id, n=n * 5, max_length=max_length, temperature=temperature,
                                   beam_size=beam_size, cultures=[language])
    else:
        blend_id = language_to_id_filtered.get(blend_language, None)

        if blend_id is None:
            raise HTTPException(status_code=404, detail="Item not found")

        labels = torch.tensor([language_id, blend_id], dtype=torch.long, device=device)
        embeddings = model.decoder.culture_embedding(labels)
        language_embedding = embeddings[0]
        blend_embedding = embeddings[1]

        if weight is None:
            weights = [0.5, 0.5]
        else:
            weights = [weight, 1 - weight]

        fictional_embedding = language_embedding * weights[0] + blend_embedding * weights[1]

        if number is not None:
            names = model.generate(culture_embedding=fictional_embedding, n=number * 5, max_length=max_length,
                                   temperature=temperature, beam_size=beam_size)
        else:
            names = model.generate(culture_embedding=fictional_embedding, n=n * 5, max_length=max_length,
                                   temperature=temperature, beam_size=beam_size)

    ngram2_probs = [ngram2.sequence_log_probability((name, language)) for name in names]
    ngram3_probs = [ngram3.sequence_log_probability((name, language)) for name in names]
    ngram4_probs = [ngram4.sequence_log_probability((name, language)) for name in names]

    scored_names = [[names[idx], ngram2_probs[idx] + ngram3_probs[idx] + ngram4_probs[idx]] for idx, name in
                    enumerate(names)]

    if blend_language is not None and language != blend_language:
        ngram2_probs_blend = [ngram2.sequence_log_probability((name, blend_language)) for name in names]
        ngram3_probs_blend = [ngram3.sequence_log_probability((name, blend_language)) for name in names]
        ngram4_probs_blend = [ngram4.sequence_log_probability((name, blend_language)) for name in names]

        blend_scores = [
            lp2 + lp3 + lp4
            for lp2, lp3, lp4 in zip(ngram2_probs_blend, ngram3_probs_blend, ngram4_probs_blend)
        ]

        scored_names = [[name, score * weights[0] + blend_scores[idx] * weights[1]] for
                        idx, (name, score) in enumerate(scored_names)]

    scored_names.sort(key=lambda item: item[1], reverse=True)

    top_names = [name for name, score in scored_names[:number if number is not None else n]]

    return {"names": top_names}
