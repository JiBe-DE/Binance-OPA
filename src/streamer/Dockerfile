FROM alpine:latest
RUN apk update && apk add python3 && apk add py3-pip
COPY . /home/project/streamer
WORKDIR /home/project/streamer
RUN pip install -r /home/project/streamer/requirements.txt
CMD python3 streamer.py