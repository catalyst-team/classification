FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libglib2.0-0 \
	libturbojpeg \
	software-properties-common \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements/requirements_min.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt --no-cache-dir

CMD mkdir -p /workspace
WORKDIR /workspace
