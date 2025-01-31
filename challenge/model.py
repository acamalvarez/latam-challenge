"""Model to predict flights delay."""

from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from challenge.variables import TOP_10_FEATURES


def get_period_day(date: str) -> str:
    """Returns perdiod date based on a date string."""
    date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
    morning_min = datetime.strptime("05:00", "%H:%M").time()
    morning_max = datetime.strptime("11:59", "%H:%M").time()
    afternoon_min = datetime.strptime("12:00", "%H:%M").time()
    afternoon_max = datetime.strptime("18:59", "%H:%M").time()

    if morning_min <= date_time <= morning_max:
        return "mañana"
    if afternoon_min <= date_time <= afternoon_max:
        return "tarde"
    return "noche"


def is_high_season(date: str) -> int:
    """Determines if the associated date is a high season date."""
    fecha_anho = int(date.split("-")[0])
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_anho)
    range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_anho)
    range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_anho)
    range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_anho)
    range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_anho)
    range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_anho)
    range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_anho)
    range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_anho)

    if (
        range1_min <= date <= range1_max
        or range2_min <= date <= range2_max
        or range3_min <= date <= range3_max
        or range4_min <= date <= range4_max
    ):
        return 1
    return 0


def get_min_diff(data: pd.DataFrame) -> float:
    """Returns the difference in dates `Fecha-0` and `Fecha-I`"""
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


class DelayModel:
    """Model to predict flights delay."""

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data["period_day"] = data["Fecha-I"].apply(get_period_day)
        data["high_season"] = data["Fecha-I"].apply(is_high_season)
        data["min_diff"] = data.apply(get_min_diff, axis=1)
        threshold_in_minutes = 15
        data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        if target_column:
            target = data[["delay"]].copy()

            return features[TOP_10_FEATURES], target

        return features[TOP_10_FEATURES]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        x_train, _, y_train, _ = train_test_split(
            features,
            target["delay"],
            test_size=0.33,
            random_state=42,
        )

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])

        reg_model_2 = LogisticRegression(
            class_weight={
                1: n_y0 / len(y_train),
                0: n_y1 / len(y_train),
            }
        )
        reg_model_2.fit(x_train, y_train)

        self._model = reg_model_2

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

        return self._model.predict(features).tolist()
