echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "LIBRARY_PATH=${LIBRARY_PATH}"
# Link against stubs when compiling
export CMAKE_LIBRARY_PATH=${LIBRARY_PATH}:${CMAKE_LIBRARY_PATH}
git clone https://github.com/apache/arrow.git
cd arrow/cpp
git checkout apache-arrow-3.0.0
mkdir release
cd release
cmake -DARROW_CUDA=${ARROW_CUDA:-OFF}  ..
make -j
sudo make install