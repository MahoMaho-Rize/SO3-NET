import os
import pprint

pp = pprint.PrettyPrinter()
from datetime import datetime

from model import Model
from config import opts


if __name__ == "__main__":
    model = Model(opts)
    if opts.network == "equivariant":
        model.train_equivariant()
    else:
        model.train()
