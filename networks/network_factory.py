import torch.nn as nn

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'APNet':
            from .networks import APNet
            network = APNet(*args, **kwargs)
        elif network_name == 'RigNetEncoder':
            from .networks import RigNetEncoder
            network = RigNetEncoder(*args, **kwargs)
        elif network_name == 'RigNetDecoder':
            from .networks import RigNetDecoder
            network = RigNetDecoder(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name
    
