yum -y update \
    && yum -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && conda clean --all --yes \
    && rpm -e --nodeps curl bzip2 \
    && yum clean all

export CONDA_ALWAYS_YES="true"

conda create -q -n p36 python=3.6 setuptools ninja
conda create -q -n p37 python=3.7 setuptools ninja
conda create -q -n p38 python=3.8 setuptools ninja
conda create -q -n p39 python=3.9 setuptools ninja
conda init
unset CONDA_ALWAYS_YES 
