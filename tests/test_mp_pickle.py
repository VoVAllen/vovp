# -*- coding: utf-8 -*-
import vovp
import torch as th
import multiprocessing as mp

def subprocess_client(tensor):
    print(tensor)
    del tensor
    # vovp.get_client().list()


def test_basic_client():
    client = vovp.init_client("/tmp/dgl_socket")
    vovp.init_reduction()
    # vovp.
    a = th.tensor([[1, 2, 3], [5, 4, 6]], device="cuda")
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    barrier = ctx.Barrier()
    p = ctx.Process(target=subprocess_client, args=(a, barrier))
    p.start()
    del a
    p.join()
    # ret_list = queue.get()
    # for ret in ret_list:
    #     assert ret

if __name__=="__main__":
    test_basic_client()
