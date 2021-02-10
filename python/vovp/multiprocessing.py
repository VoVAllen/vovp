import os
import threading
import multiprocessing
from multiprocessing.util import register_after_fork
from multiprocessing.reduction import ForkingPickler
import numpy as np
import torch
from .client import get_client

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class PlasmaIDHolder:
    def __init__(self, object_id) -> None:
        self.object_id = object_id

def rebuild_tensor(object_id):
    client = get_client()
    return client.get_tensor(object_id)

def reduce_tensor(tensor):
    # print("Get client")
    client = get_client()

    object_id = id_generator(10)
    client.put_tensor(object_id, tensor)

    # print("Put Tensor")
    return (rebuild_tensor, (object_id,))

def init_reduction():
    ForkingPickler.register(torch.Tensor, reduce_tensor)