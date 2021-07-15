import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ds.config import Config
from ds.utils import encode_data, fit_model, get_age_bucket

if __name__ == "__main__":

    config = Config()

    logging.basicConfig(level=config.logging_level)

    data = pd.read_csv(
        Path(config.data_folder) / config.project / "train.csv", index_col=0
    )

    data_cleaned = data.dropna(subset=["Age", "Sex"]).copy()

    logging.debug(
        f" Number of dropped rows: {data.shape[0] - data_cleaned.shape[0]} / {data.shape[0]}"
    )

    data_cleaned["Age_Buckets"] = data_cleaned["Age"].map(get_age_bucket)
    data_cleaned["Sex"]

    Y = data_cleaned["Fare"].values
    X_1 = data_cleaned["Age"].values
    X_2, mapping_X_2 = encode_data(data_cleaned["Sex"].values)
    X = np.column_stack([X_1, X_2])

    beta, residual = fit_model(X, Y)

    # TODO:
    # 1. Compute significance
    # 2. Model with categorical variables (Is this the way to go about it?)
    # 3. Fare seems to be related to Sex and Age
    # 4. Is Fare related to Survival?

    print(beta)

    print(residual)

    print(mapping_X_2)

    predictions = X.dot(beta)

    zoom = [0, 3000]
    plt.plot(predictions[zoom[0] : zoom[1]], label="predictions")
    plt.plot(Y[zoom[0] : zoom[1]], label="values")
    plt.show()
