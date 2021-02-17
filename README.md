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
import torch as th
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
Install arrow library with CUDA support
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

## Pros and Cons comparing to current DGL solution
### Pros
- Clear reference counting semantic (no more worries on the lifetime management)
- Only name is needed when get the tensor (current DGL needs shape and dtype to reconstruct shared-memory tensor)
- Support CUDA tensor (which is useful for DistGPUGraph)
- Neat interface
- Can support [huge pages](https://arrow.apache.org/docs/python/plasma.html?highlight=hugepages#using-plasma-with-huge-pages)
- Multi-thread memcopy will be used when memory size > 1MB

### Cons
- Need to start a seperate process (Can be a pros since it can live longer than DGL training process)
- New dependency on arrow and arrow-cuda (Can be solved by static linking?)