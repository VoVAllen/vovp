FROM nvidia/cuda:11.2.1-cudnn8-devel-centos7

RUN yum -y update

COPY install/centos_install_core.sh /install/centos_install_core.sh
RUN bash /install/centos_install_core.sh

COPY install/centos_install_build.sh /install/centos_install_build.sh
RUN bash /install/centos_install_build.sh

COPY install/centos_install_gcc7.sh /install/centos_install_gcc7.sh
RUN bash /install/centos_install_gcc7.sh

COPY install/centos_install_gflags.sh /install/centos_install_gflags.sh
RUN bash /install/centos_install_gflags.sh

ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV ARROW_CUDA=ON
ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64/stubs

COPY install/centos_install_arrow.sh install/centos_install_arrow.sh
RUN bash /install/centos_install_arrow.sh

COPY install/centos_install_python.sh install/centos_install_python.sh
RUN bash /install/centos_install_python.sh

CMD [ "/bin/bash" ]