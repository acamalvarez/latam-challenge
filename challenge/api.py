import os
from enum import Enum
from typing import List

import fastapi
import numpy as np
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel
from challenge.variables import TOP_10_FEATURES


class TipoVuelo(Enum):
    """Enumeration for TipoVuelo."""
    I = "I"
    N = "N"


class Flight(BaseModel):
    """Base model to represent a flight."""
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightData(BaseModel):
    """Base model to represent a list of Flights"""
    flights: List[Flight]


app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    absolute_path = os.path.dirname(__file__)
    relative_path = "../data/data.csv"
    full_path = os.path.join(absolute_path, relative_path)
    data = pd.read_csv(full_path)
    model = DelayModel()
    features, target = model.preprocess(data, "delay")
    model.fit(features, target)

    arilines = data["OPERA"].unique()

    try:
        data = pd.DataFrame(
            0, index=np.arange(len(flight_data.flights)), columns=TOP_10_FEATURES
        )
        for index, flight in enumerate(flight_data.flights):
            if flight.OPERA not in arilines:
                raise ValueError("A aerolínea no existe en los datos.")

            if flight.OPERA in TOP_10_FEATURES:
                data.loc[index]["OPERA" + flight.OPERA] = 1

            if flight.TIPOVUELO not in TipoVuelo.__members__:
                values = list((member.value for member in TipoVuelo))
                raise ValueError(f"Tipo vuelo debe estar en {values}.")
            data.loc[index]["TIPOVUELO_I"] = int(flight.TIPOVUELO == "I")

            if flight.MES not in range(1, 13):
                raise ValueError("El mes debe ser entre 1 y 12.")
            month = "MES_" + str(flight.MES)

            if month in TOP_10_FEATURES:
                data.loc[index][month] = 1

        prediction = model.predict(data)

        return {"predict": prediction}

    except Exception as e:
        return {"status_code": 400, "message": str(e)}
