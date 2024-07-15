FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="bigscience-workshop"
LABEL repository="petals"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
    wget \
      git \
        && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
  ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python~=3.10.12 pip && \
    pip install --no-cache-dir "torch>=1.12" && \
    conda clean --all && rm -rf ~/.cache/pip

VOLUME /root/.cache

COPY . chat.petals.dev/
COPY kaggle.json /root/.kaggle/
WORKDIR /home/chat.petals.dev
RUN mkdir -p SOURCE_DOCUMENTS
VOLUME /home/chat.petals.dev/SOURCE_DOCUMENTS
RUN pip install -r requirements.rag.txt

CMD ["flask", "run","--host","0.0.0.0","--port","8181"]