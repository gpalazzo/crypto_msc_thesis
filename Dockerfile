ARG BASE_IMAGE=python:3.8-buster
FROM $BASE_IMAGE

COPY --from=python:3.8-buster / /

ENV PIP_NO_CACHE_DIR=1

# install project requirements
COPY pyproject.toml .
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    poetry config virtualenvs.create false &&\
    poetry lock &&\
    poetry install &&\
    rm -f requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
WORKDIR /home/kedro
COPY . .
RUN chown -R kedro:${KEDRO_GID} /home/kedro

#testar com isso comentado depois!
USER kedro

RUN chmod -R a+w /home/kedro

ENV PYTHONPATH="/home/kedro/src:${PYTHONPATH}"

# keep Kedro running for dev cycle
# altere seu token aqui
# caso prefira jupyter notebook ao invés do jupyter lab, troque `lab` por `notebook`
CMD ["kedro", "jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--NotebookApp.token='mscthesis'"]