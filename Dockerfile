FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y tzdata wget   git && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
RUN apt install -y libjpeg-turbo8 libtiff-dev libgl1
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN  conda install -n base -c conda-forge opencv 'libjpeg-turbo=2.1.5' -y