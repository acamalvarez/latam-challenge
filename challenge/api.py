import os
from enum import Enum
from typing import List

import fastapi
import numpy as np
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel


class TipoVuelo(Enum):
    I = "I"
    N = "N"


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightData(BaseModel):
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
    top_10_features = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    try:
        data = pd.DataFrame(
            0, index=np.arange(len(flight_data.flights)), columns=top_10_features
        )
        for index, flight in enumerate(flight_data.flights):
            if flight.OPERA not in arilines:
                raise ValueError("A aerolínea no existe en los datos.")

            else:
                if flight.OPERA in top_10_features:
                    data.loc[index]["OPERA" + flight.OPERA] = 1

            if flight.TIPOVUELO not in TipoVuelo.__members__:
                values = list((member.value for member in TipoVuelo))
                raise ValueError(f"Tipo vuelo debe estar en {values}.")
            data.loc[index]["TIPOVUELO_I"] = int(flight.TIPOVUELO == "I")

            if flight.MES not in range(1, 13):
                raise ValueError("El mes debe ser entre 1 y 12.")
            month = "MES_" + str(flight.MES)

            if month in top_10_features:
                data.loc[index][month] = 1

        prediction = model.predict(data)

        return {"predict": prediction}

    except Exception as e:
        return {"status_code": 400, "message": str(e)}
