FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update && apt upgrade -y 
RUN mkdir /app
# COPY ./ /app
RUN pip install -U pip
RUN pip install torchaudio
RUN pip install diffusers
RUN pip install vector_quantize_pytorch