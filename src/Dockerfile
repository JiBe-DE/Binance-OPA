FROM python:3.8.4-slim-buster
COPY . /home/project/
WORKDIR /home/project/
RUN pip install -r /home/project/requirements.txt
CMD uvicorn binance_api:api --reload --host 0.0.0.0