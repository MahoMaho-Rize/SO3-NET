import os
from model import Model
from config import opts


if __name__ == "__main__":
    model = Model(opts)
    if opts.network == "equivariant":
        model.test_equivariant()
    else:
        model.test()
