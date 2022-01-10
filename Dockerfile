FROM nvcr.io/nvidia/pytorch:21.02-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing

# RUN useradd -m tan

# RUN chown -R tan:tan /home/tan/

# COPY --chown=tan . /home/tan/budget_bert/

COPY . /home/tan/budget_bert/

# USER tan

RUN cd /home/tan/budget_bert/ && pip3 install -r requirements.txt

# RUN cd /home/tan/budget_bert/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /home/tan/budget_bert/