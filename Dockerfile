FROM nvcr.io/nvidia/pytorch:20.10-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing

RUN useradd -m tan

RUN chown -R tan:tan /home/tan/

COPY --chown=tan . /home/tan/budget_bert/

USER tan

RUN cd /home/tan/budget_bert/ && pip install -r requirements.txt

WORKDIR /home/tan/budget_bert/
