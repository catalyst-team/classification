FROM catalystteam/catalyst:20.01-fp16

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

COPY requirements/requirements-docker.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt --no-cache-dir

CMD mkdir -p /workspace
WORKDIR /workspace
