#syntax=docker/dockerfile-upstream:master-experimental

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-11.html#rel_11-11
# PyTorch 1.11.0a0+b6df043, Python 3.8.12, CUDA 11.5.0, Ubuntu 20.04
# FROM nvcr.io/nvidia/pytorch:21.11-py3
FROM pytorch/pytorch
# FROM nvidia/cuda:11.5.2-base-ubuntu20.04
RUN  --mount=type=cache,target=/var/cache \
    DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates &&\
    sed -i "s/http:\/\/archive\.ubuntu\.com\/ubuntu\//https:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//g" /etc/apt/sources.list && \
    sed -i "s/http:\/\/security\.ubuntu\.com\/ubuntu\//https:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//g" /etc/apt/sources.list && \
    cat /etc/apt/sources.list && echo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        locales tzdata lsb-release iputils-ping \
        apt-utils apt-transport-https gnupg dirmngr openssl software-properties-common  \
        tar wget ssh git mercurial vim openssh-client psmisc rsync \
        build-essential autoconf libtool \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        libnlopt-dev libpq-dev libffi-dev libcairo-dev libedit-dev \
        libcurl4-nss-dev libsasl2-dev libsasl2-modules libapr1-dev libsvn-dev \
        libjpeg-dev htop sudo liblapack-dev libatlas-base-dev ssh \
        graphviz libgraphviz-dev curl direnv jq libgl1\
    && \
    (printf 'eval "$(direnv hook bash)"' >> ~/.bashrc) && \
    (echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config) && \
    (echo 'root:root' | chpasswd) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG pip_arg="-i https://pypi.tuna.tsinghua.edu.cn/simple"

ADD requirements-dev.txt /tmp/requirements-dev.txt
# not update the installed package
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && \
    pip install -r /tmp/requirements-dev.txt ${pip_arg}

RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/opt/conda/pkgs \
    pip install -U pip &&  \
    pip install -U -r /tmp/requirements.txt ${pip_arg}