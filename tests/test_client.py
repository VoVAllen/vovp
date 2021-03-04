# -*- coding: utf-8 -*-
import vovp
import torch as th


def test_basic_client():
    client = vovp.init_client("/tmp/dgl_socket")
    a = th.tensor([[1, 2, 3], [5, 4, 6]])
    ret_a = client.put_tensor("test111", a)
    ret_b = client.get_tensor('test111')
    ret_c = client.get_tensor('test111')
    assert th.equal(ret_a, ret_b)
    ret_a[0][0] = 999
    assert th.equal(ret_a, ret_b)
    del ret_b, ret_c, ret_a, a
    client.list()


test_basic_client()
