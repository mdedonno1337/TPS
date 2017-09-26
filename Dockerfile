FROM debian
MAINTAINER Marco De Donno <Marco.DeDonno@unil.ch>

RUN apt update && \
    apt upgrade -y

################################################################################
###   Python

RUN apt install -y python python-pip python-dev \
    build-essential libssl-dev libffi-dev libpq-dev
RUN pip install --upgrade pip

RUN pip install numpy \
    pillow \
    scipy

RUN pip install cython

################################################################################
###   TPS library

COPY ./TPS /TPS/TPS

RUN echo /TPS/ > /usr/local/lib/python2.7/dist-packages/mdedonno.pth

WORKDIR /TPS

RUN make -C /TPS/TPS/TPSCy

################################################################################
###   Unit test by default

COPY TPSModules_unittest.py /TPS/TPSModules_unittest.py
RUN python /TPS/TPSModules_unittest.py
