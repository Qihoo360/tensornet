FROM tensorflow/tensorflow:2.4.0

RUN apt-get update

COPY ./tools/install-dependencies.sh .
RUN bash ./install-dependencies.sh

# you can download the package in advance, and use COPY instead
#COPY bazel-3.7.2-linux-x86_64 .
RUN wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-linux-x86_64
RUN mv bazel-3.7.2-linux-x86_64 /usr/local/bin/bazel && \
        chmod a+x /usr/local/bin/bazel

#COPY openmpi-1.4.5.tar.gz .
RUN wget https://download.open-mpi.org/release/open-mpi/v1.4/openmpi-1.4.5.tar.gz
RUN  mkdir -p /opt && \
    mv openmpi-1.4.5.tar.gz /opt && \
    cd /opt && tar -zxf openmpi-1.4.5.tar.gz && \
    cd /opt/openmpi-1.4.5 && \
    ./configure CFLAGS="-fPIC" CXXFlAGS="-fPIC" --prefix=/opt/openmpi --enable-static && \
    make -j20 && \
    make install

ENV PATH "/opt/openmpi/bin:${PATH}"
ENV PYTHONPATH "/tensornet:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/opt/openmpi/lib:${LD_LIBRARY_PATH}"
