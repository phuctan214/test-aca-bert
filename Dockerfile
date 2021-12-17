FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m tan

RUN chown -R tan:tan /home/tan/

COPY --chown=tan . /home/tan/budget_bert/

USER tan

RUN cd /home/tan/budget_bert/ && pip3 install -r requirements.txt

WORKDIR /home/tan/budget_bert/