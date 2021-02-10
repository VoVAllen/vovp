# -*- coding: utf-8 -*-
import vovp
import torch as th
import multiprocessing as mp

def subprocess_client(queue):
    import vovp, logging
    import torch as th
    
    try:
        results = []
        client = vovp.init_client("/tmp/dgl_socket")
        ret_sub = client.get_tensor('asdasd')
        cmp_a = th.tensor([[1, 2, 3], [5, 4, 6]])
        results.append(th.equal(ret_sub, cmp_a))
        queue.put(results)
    except Exception as e:
        logging.info('General exception noted.', exc_info=True)
        queue.put(e)


def test_basic_client():
    client = vovp.init_client("/tmp/dgl_socket")
    a = th.tensor([[1, 2, 3], [5, 4, 6]])
    ret_a = client.put_tensor("asdasd", a)
    ret_b = client.get_tensor('asdasd')
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=subprocess_client, args=(queue, ))
    p.start()
    p.join()
    ret_list = queue.get()
    for ret in ret_list:
        assert ret

if __name__=="__main__":
    test_basic_client()
