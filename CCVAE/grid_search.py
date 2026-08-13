import argparse

from CCVAE.evaluate import evaluate
from CCVAE.train import train


def grid_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    lr = args.lr
    model, train_names = train(lr)
    evaluate(model, lr, train_names)


if __name__ == "__main__":
    grid_search()
