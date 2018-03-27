FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir -p /app /app/data /app/models
COPY sandbox_go /app/sandbox_go
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY Makefile /app/Makefile

RUN cd /app && make build

WORKDIR /app
ENTRYPOINT [ "/usr/bin/python", "-m", "sandbox_go" ]
