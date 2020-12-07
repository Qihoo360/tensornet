FROM ubuntu:18.04

RUN apt-get update && apt-get install wget -y && \
    apt-get install python3.7 -y && \
    apt-get install python3-pip -y && \
    apt-get install git -y && \
    apt-get install libssl-dev -y && \
    apt-get install lib32z1-dev -y && \
    apt-get install curl -y

RUN wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-linux-x86_64 && \
    mv bazel-3.1.0-linux-x86_64 /usr/local/bin/bazel && \
    chmod a+x /usr/local/bin/bazel

RUN pip3 install --upgrade pip && \
    pip3 install tensorflow==2.3.0

RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN wget https://download.open-mpi.org/release/open-mpi/v1.4/openmpi-1.4.5.tar.gz && \
    mkdir -p /root/opt && \
    tar -zxf openmpi-1.4.5.tar.gz -C /root/opt/ && \
    mv /root/opt/openmpi-1.4.5 /root/opt/openmpi && \
    cd /root/opt/openmpi && \
    ./configure CFLAGS="-fPIC" CXXFlAGS="-fPIC" --prefix=/root/opt/openmpi --enable-static && \
    make -j20 && \
    make install

RUN git clone https://github.com/Qihoo360/tensornet.git && \
    cd /tensornet && \
    bash configure.sh --openmpi_path /root/opt/openmpi && \
    bazel build -c opt //core:_pywrap_tn.so && \
    cp -f /tensornet/bazel-bin/core/_pywrap_tn.so /tensornet/tensornet/core

ENV PATH "/root/opt/openmpi/bin:${PATH}"
ENV PYTHONPATH "/tensornet:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/root/opt/openmpi/lib:${LD_LIBRARY_PATH}"

CMD ["python", "-c", "import tensorflow as tf; import tensornet as tn; print(tn.version)"]
