FROM nvidia/cuda:10.2-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 wget \
        build-essential openjdk-8-jdk
ENV PATH="/opt/conda/bin:${PATH}"
ENV JCC_JDK=/usr/lib/jvm/java-1.8.0-openjdk-$(dpkg --print-architecture)
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash Miniconda3-latest-Linux-x86_64.sh -p /opt/conda -b && \
        rm Miniconda3-latest-Linux-x86_64.sh
RUN pip3 install \
        jcc==3.10 \
        kornia==0.5.6 \
        torch==1.9.1 \
        scipy==1.6.3 \
        nibabel==3.2.1 \
        matplotlib==3.4.2 \
        tqdm==4.51.0 \
        improc3d==0.5.2 \
        opencv-python==4.5.3.56 \
        git+https://gitlab.com/shan-utils/resize.git@0.1.1
RUN git clone https://github.com/nighres/nighres
WORKDIR /tmp/nighres
RUN ./build.sh && pip install . && rm -rf /tmp/nighres

COPY scripts /tmp/scripts
COPY deep_meninges /tmp/deep_meninges
COPY setup.py README.md /tmp/
WORKDIR /tmp
RUN pip install . && rm -rf /tmp/*
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
