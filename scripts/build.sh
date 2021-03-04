ARROW_CUDA=${ARROW_CUDA:-OFF}
cd /io
mkdir -p build_docker
cd /io/third_party/Plasmastore
# rm build_docker -rf
mkdir -p build_docker
cd build_docker
export CXX=/opt/rh/devtoolset-7/root/usr/bin/g++
export CC=/opt/rh/devtoolset-7/root/usr/bin/gcc

if [ "${ARROW_CUDA}" == "ON" ]; then
    cp /usr/local/lib64/libarrow_cuda.so.300.0.0 /io/build_docker/libarrow_cuda.so.300
    patchelf --set-rpath "./" /io/build_docker/libarrow_cuda.so.300
    # Link against CUDA stubs when compiling
    export CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:${LIBRARY_PATH}:${CMAKE_LIBRARY_PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:${LIBRARY_PATH}:${LD_LIBRARY_PATH}:/usr/local/lib
    cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
fi

cmake -DARROW_CUDA=${ARROW_CUDA} ..
make -j
echo "ARROW_CUDA=${ARROW_CUDA}"
cp /usr/local/lib64/libarrow.so.300.0.0 /io/build_docker/libarrow.so.300
cp /io/third_party/Plasmastore/build_docker/libplasma_client.so /io/build_docker/libplasma_client.so
cp /io/third_party/Plasmastore/build_docker/plasma-store-server /io/build_docker/plasma-store-server

# export LD_LIBRARY_PATH=/io/third_party/Plasmastore/build_docker/:${LD_LIBRARY_PATH}
# ls /io/third_party/Plasmastore/build_docker/
echo "LD:${LD_LIBRARY_PATH}"
# cd /io/build_docker
# cmake -DVOVP_CUDA=${ARROW_CUDA} ..
# make -j
cd /io
# source /opt/conda/etc/profile.d/conda.sh
source activate
for env in p36 p37 p38 p39; do
conda activate $env
python3 setup.py bdist_wheel --plat-name=manylinux2014_x86_64
done
# python3 setup.py sdist