FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
RUN pip install scipy==1.6.3 \
        nibabel==3.2.1 \
        matplotlib==3.4.2 \
        tqdm==4.51.0 \
        improc3d==0.5.2 \
        opencv-python==4.5.3.56 \
        git+https://gitlab.com/shan-utils/resize.git@0.1.1
COPY scripts /tmp/scripts
COPY ct_synth /tmp/ct_synth
COPY setup.py README.md /tmp/
WORKDIR /tmp
RUN pip install . && rm -rf /tmp/*
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["bash"]
