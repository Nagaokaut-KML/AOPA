FROM ubuntu:18.04

RUN apt-get -qq update && apt install -y --no-install-recommends build-essential sudo

RUN apt-get update && apt-get install -y \
libsm6 \
  apt-utils \
	python3 \ 
  python3-pip \
	wget \ 
	make \
	file \ 
	unzip \
	git \ 
	curl \
	nodejs \
  xz-utils\
  sudo \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libxft-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pyzmq --install-option="--zmq=bundled" && \
    pip3 install --upgrade jupyterlab && \
    pip3 install --upgrade \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pandas \
    gensim \
    tqdm  \
    opencv-python

# LAPACK/BLAS (scikit-learnで必要)
RUN cd /tmp \
    && wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz \
    && tar zxf lapack-3.8.0.tar.gz \
    && cd lapack-3.8.0/ \
    && cp make.inc.example make.inc \
    && make blaslib \
    && make lapacklib \
    && cp librefblas.a /usr/lib/libblas.a \
    && cp liblapack.a /usr/lib/liblapack.a \
    && cd / \
    && rm -rf /tmp/*

# machine learning library
RUN pip3 install tensorflow \
	 keras 


RUN pip3 install torchvision

