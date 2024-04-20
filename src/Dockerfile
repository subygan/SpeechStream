FROM python:3.11-bullseye

RUN apt update && apt install -y git ffmpeg redis-server && apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install nvidia-pyindex

RUN set -ex
RUN pip install --upgrade pip setuptools wheel cython && pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/


COPY . .

RUN wget -q https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephonic/versions/1.0.1/files/diar_msdd_telephonic.nemo -O /app/diar_msdd_telephonic.nemo
RUN wget -q https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_multilingual_marblenet/versions/1.10.0/files/vad_multilingual_marblenet.nemo -O /app/vad_multilingual_marblenet.nemo


# Expose the port on which the Flask app will run
EXPOSE 5000

CMD ["sleep", "infinity"]