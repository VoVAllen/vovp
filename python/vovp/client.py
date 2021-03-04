import _vovp
import torch as th
from torch.utils.dlpack import to_dlpack, from_dlpack


class VovpClient:
    def __init__(self, socket_name):
        self.plasma_client = _vovp.VovpPlasmaClient(socket_name)

    def put_tensor(self, object_id, tensor):
        new_dlp = self.plasma_client.put_tensor(to_dlpack(tensor), object_id, False, True)
        return from_dlpack(new_dlp)

    def get_tensor(self, object_id):
        new_dlp = self.plasma_client.get_tensor(object_id)
        return from_dlpack(new_dlp)
    
    def list(self):
        self.plasma_client.list()


VOVP_CLIENT = None


def init_client(socket_name="/tmp/dgl_socket"):
    global VOVP_CLIENT
    VOVP_CLIENT = VovpClient(socket_name)
    return VOVP_CLIENT


def get_client():
    global VOVP_CLIENT
    if (VOVP_CLIENT is None):
        init_client()
    return VOVP_CLIENT
