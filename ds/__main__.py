#!/usr/bin/env python3
"""Entry point"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ds.config import Config
from ds.utils import (
    encode_data,
    fit_model,
    get_age_bucket,
    r_squared,
    residual_sum_of_squares,
    total_sum_of_squares,
)

if __name__ == "__main__":

    config = Config()

    logging.basicConfig(level=config.logging_level)

    data = pd.read_csv(
        Path(config.data_folder) / config.project / "train.csv", index_col=0
    )

    data_cleaned = data.dropna(subset=["Age", "Sex"]).copy()

    logging.debug("Number of dropped rows %i", data.shape[0] - data_cleaned.shape[0])

    data_cleaned["Age_Buckets"] = data_cleaned["Age"].map(get_age_bucket)

    Y = data_cleaned["Fare"].values
    X_1 = data_cleaned["Age"].values
    X_2, mapping_X_2 = encode_data(data_cleaned["Sex"].values)
    X = np.column_stack([X_1, X_2])

    beta, residual = fit_model(X, Y)

    print(beta)

    print(residual)

    print(mapping_X_2)

    predictions = X.dot(beta)

    print(total_sum_of_squares(Y))
    print(residual_sum_of_squares(Y, predictions))

    print(r_squared(Y, predictions))

    # zoom = [0, 3000]
    # plt.plot(predictions[zoom[0] : zoom[1]], label="predictions")
    # plt.plot(Y[zoom[0] : zoom[1]], label="values")
    # plt.show()
