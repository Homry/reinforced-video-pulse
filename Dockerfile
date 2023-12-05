FROM tensorflow/tensorflow:2.14.0-gpu
LABEL authors="Homry"

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


WORKDIR pulse
COPY data data
COPY src src
COPY videos videos
COPY main.py main.py

CMD ["bash"]