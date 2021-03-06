import os

from factorizer import MatrixFactorizer
import pandas as pd
import numpy as np
import fire
from datetime import datetime
from uuid import uuid4


def prepare_data(path):
    """ Implement as needed """
    pass


def fit_model(path, p, alpha, beta, n_iter):
    event_id = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    start = datetime.now()
    X = prepare_data(path)

    mf = MatrixFactorizer(p, alpha, beta)
    mf.fit(X, n_iter)
    end = datetime.now()

    if not os.path.exists(f"./models/{event_id}"):
        os.makedirs(f"./models/{event_id}")

    with open(f"models/{event_id}/u.npy", "wb") as u_file, open(f"models/{event_id}/v.npy", "wb") as v_file:
        np.save(u_file, mf.u)
        np.save(v_file, mf.v)

    print(f"Training took: {(end - start).seconds}s")
    print(f"Model trained and saved in ./models with uuid:\n{event_id}")


def main():
    fire.Fire(fit_model)


if __name__ == "__main__":
    main()