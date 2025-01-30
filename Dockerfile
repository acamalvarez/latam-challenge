# syntax=docker/dockerfile:1.2
FROM python:latest

WORKDIR /app

COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .

COPY . ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt

EXPOSE 8080

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
