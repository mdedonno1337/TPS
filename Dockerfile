FROM debian:10
LABEL maintainer "Marco De Donno <Marco.DeDonno@unil.ch>"

RUN apt-get update && \
    apt-get install -y python python-pip python-dev build-essential libssl-dev libffi-dev libpq-dev

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

################################################################################
###   TPS library

COPY ./TPS /TPS/TPS

RUN echo /TPS/ > /usr/local/lib/python2.7/dist-packages/mdedonno.pth

WORKDIR /TPS

RUN make -C /TPS/TPS/TPSCy

################################################################################
###   Unit test by default

COPY doctester.py /TPS/doctester.py
RUN python /TPS/doctester.py

ADD coverage.sh /TPS/coverage.sh
RUN chmod +x /TPS/coverage.sh
