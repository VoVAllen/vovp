# Vovp
Key-Value(shared-memory tensor/CUDA tensor) storage with zero-copy

## Example
### Start server
```bash
# bash
# start server listening at "/tmp/dgl_socket" with 1GB shared memory limit 
plasma-store-server -m 1000000000 -s "/tmp/dgl_socket"
```
### Client usage (on the same machine)
#### Use KV store
In one processs
```python
import vovp
client = vovp.init_client("/tmp/dgl_socket")
a = th.tensor([[1, 2, 3], [5, 4, 6]]) # a can be a CUDA tensor
# This will ask server to create the same tensor, copy from this tensor
# and return the tensor which underlying memory is hosted by the server process
ret_a = client.put_tensor("test111", a) # ret_a has same value with a, but holding memory allocated by server process 
```
In another process
```python
import vovp
client = vovp.init_client("/tmp/dgl_socket")
ret_a2 = client.get_tensor("test111") # Exact same underlying memory with ret_a
```

#### Direct use KV store for pickling between process
```python
import vovp
client = vovp.init_client("/tmp/dgl_socket")
vovp.init_reduction()
# now when you pass tensor between process, it will put the tensor into kv store and get tensor in another process
```

## Installation
### Prerequesite
Install arrow library with CUDA support`
```bash
# Install system packages listed in https://arrow.apache.org/docs/developers/cpp/building.html if needed
git clone https://github.com/apache/arrow.git
cd arrow/cpp
mkdir release
cd release
cmake -DARROW_CUDA=ON -DARROW_PLASMA=ON ..
make -j
sudo make install
```
### Install Vovp
```bash
git clone --recursive https://github.com/VoVAllen/vovp.gitc
cd vovp
pip install .
```
