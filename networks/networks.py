import torch
import torch.nn as nn
from .network_factory import NetworkBase
from utils.util import parse_styles

class APNet(NetworkBase):

    """
    Attribute Prediction Networkn: This parameter regressor is a function F that maps a latent code w to a vector of semantic control
    parameters pw = F(w). In practice, we model F using a Multi-MLP Network with ELU activations after every intermediate layers.
    """

    def __init__(self, in_dim=9088, out_dim=257): #257 = 80(shape)+64(expression)+80(albedo)+3(rotation)+27(lighting)+3(translation)
        super(APNet, self).__init__()
        self._name = 'APNet'

        layers = []
        layers.append(nn.Sequential(nn.Linear(in_dim, 4096), nn.ELU()))
        layers.append(nn.Sequential(nn.Linear(4096, 2048), nn.ELU()))
        layers.append(nn.Sequential(nn.Linear(2048, 1024), nn.ELU()))
        layers.append(nn.Sequential(nn.Linear(1024, 512), nn.ELU()))
        layers.append(nn.Linear(512, out_dim))

        self.fcs = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            
    def forward(self, inp):
        #print ("((((((((")
        #print (inp)
        x = self.fcs(inp)
        #print (")))))))")
        return x


class RigNetEncoder(NetworkBase):

    def __init__(self):
        super(RigNetEncoder, self).__init__()
        self._name = 'RigNetEncoder'

        self._nets = nn.ModuleList([nn.Linear(512, 32) for i in range(15)] +
                                   [nn.Linear(256, 32) for i in range(3)] +
                                   [nn.Linear(128, 32) for i in range(3)] +
                                   [nn.Linear(64, 32) for i in range(3)] +
                                   [nn.Linear(32, 32) for i in range(2)])

    def forward(self, inp):  # inp: batch, 9088
        inp = parse_styles(inp)
        outs = []
        for ind in range(26):
            out = self._nets[ind](inp[ind])
            out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        return outs


class RigNetDecoder(NetworkBase):

    def __init__(self):
        super(RigNetDecoder, self).__init__()
        self._name = "RigNetDecoder"

        self._nets_shape_albedo = nn.ModuleList([nn.Linear(112, 256) for i in range(15)] +  # shape | albedo 80 + 32
                                       [nn.Linear(112, 128) for i in range(3)] +
                                       [nn.Linear(112, 64) for i in range(3)] +
                                       [nn.Linear(112, 64) for i in range(3)] +
                                       [nn.Linear(112, 32) for i in range(2)])
        self._nets_exp = nn.ModuleList([nn.Linear(96, 256) for i in range(15)] + # expression 64 + 32
                                       [nn.Linear(96, 128) for i in range(3)] +
                                       [nn.Linear(96, 64) for i in range(3)] +
                                       [nn.Linear(96, 64) for i in range(3)] +
                                       [nn.Linear(96, 32) for i in range(2)])
        self._nets_pose = nn.ModuleList([nn.Linear(35, 256) for i in range(15)] + # pose 3 + 32
                                        [nn.Linear(35, 128) for i in range(3)] +
                                        [nn.Linear(35, 64) for i in range(3)] +
                                        [nn.Linear(35, 64) for i in range(3)] +
                                        [nn.Linear(35, 32) for i in range(2)])
        self._nets_light = nn.ModuleList([nn.Linear(59, 256) for i in range(15)] + # light: 27 + 32
                                         [nn.Linear(59, 128) for i in range(3)] +
                                         [nn.Linear(59, 64) for i in range(3)] +
                                         [nn.Linear(59, 64) for i in range(3)] +
                                         [nn.Linear(59, 32) for i in range(2)])

        self._nets = nn.ModuleList([nn.Linear(256, 512) for i in range(15)] +
                                   [nn.Linear(128, 256) for i in range(3)] +
                                   [nn.Linear(64, 128) for i in range(3)] +
                                   [nn.Linear(64, 64) for i in range(3)] +
                                   [nn.Linear(32, 32) for i in range(2)])

    def forwardShape(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        paramsS = params[:, :80] - paramsW[:, :80]
        outs = []
        # for ind in range(26):
        # for ind in range(6):
        for ind in range(4):
            inp_cat = torch.cat((inp[:, ind], paramsS), dim=1)
            out = self._nets_shape_albedo[ind](inp_cat)  # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            outs.append(out)
        outs = torch.cat(outs, dim=1)  # batch, 9088  --- 3072
        # print ("outs:", outs.shape)
        new_latent = latent.clone()
        # Use Our Reduced StyleSpace
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048]

        return new_latent
