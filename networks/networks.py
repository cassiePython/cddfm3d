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
        x = self.fcs(inp)
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
        #self._nets256 = nn.ModuleList([nn.Linear(256, 32) for i in range(3)])
        #self._nets128 = nn.ModuleList([nn.Linear(128, 32) for i in range(3)])
        #self._nets64 = nn.ModuleList([nn.Linear(64, 32) for i in range(3)])
        #self._nets32 = nn.ModuleList([nn.Linear(32, 32) for i in range(2)])
        #for _ in range(18):
        #    self._nets.append(nn.Linear(512, 32)) 

    def forward(self, inp): # inp: batch, 9088
        #N = inp.shape[0]
        #inp = inp.view(N, 18, 512)
        inp = parse_styles(inp)
        outs = []
        for ind in range(9):# 6 for exp 2021 1/8   9 for exp 2021 1/15
        #for ind in range(17): # for pose and shape, use 4
        #for ind in range(9): # for light with RGB-right layers 
        #for ind in range(4): # for shape
            out = self._nets[ind](inp[ind])
            out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 26, 32
        return outs 

class RigNetDecoder(NetworkBase):

    def __init__(self):
        super(RigNetDecoder, self).__init__()
        self._name = "RigNetDecoder"
        
        self._nets_one = nn.ModuleList([nn.Linear(112, 256) for i in range(15)] + # shape
                                    [nn.Linear(112, 128) for i in range(3)] +
                                   [nn.Linear(112, 64) for i in range(3)] +
                                   [nn.Linear(112, 64) for i in range(3)] +
                                   [nn.Linear(112, 32) for i in range(2)])
        self._nets_exp = nn.ModuleList([nn.Linear(96, 256) for i in range(15)] +
                                        [nn.Linear(96, 128) for i in range(3)] +
                                       [nn.Linear(96, 64) for i in range(3)] +
                                       [nn.Linear(96, 64) for i in range(3)] +
                                       [nn.Linear(96, 32) for i in range(2)])
        self._nets_pose = nn.ModuleList([nn.Linear(35, 256) for i in range(15)] +
                                         [nn.Linear(35, 128) for i in range(3)] +
                                        [nn.Linear(35, 64) for i in range(3)] +
                                        [nn.Linear(35, 64) for i in range(3)] +
                                        [nn.Linear(35, 32) for i in range(2)])
        self._nets_light = nn.ModuleList([nn.Linear(59, 256) for i in range(15)] +
                                        [nn.Linear(59, 128) for i in range(3)] +
                                       [nn.Linear(59, 64) for i in range(3)] +
                                       [nn.Linear(59, 64) for i in range(3)] +
                                       [nn.Linear(59, 32) for i in range(2)])
        self._nets = nn.ModuleList([nn.Linear(256, 512) for i in range(15)] +
                                    [nn.Linear(128, 256) for i in range(3)] +
                                   [nn.Linear(64, 128) for i in range(3)] +
                                   [nn.Linear(64, 64) for i in range(3)] +
                                   [nn.Linear(32, 32) for i in range(2)])
        """
        self._nets = nn.ModuleList([nn.Linear(112, 512) for i in range(15)] +
                                    [nn.Linear(112, 256) for i in range(3)] +
                                   [nn.Linear(112, 128) for i in range(3)] +
                                   [nn.Linear(112, 64) for i in range(3)] +
                                    [nn.Linear(112, 32) for i in range(2)])
        """

    def forwardS2(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        #N = inp.shape[0]
        #latent = latent.view(N, 18, 512)
        subS = (params[:, :80] - paramsW[:, :80]) ** 2
        subS = nn.functional.sigmoid(subS)
        subA = subS * params[:, :80]
        subB = (1-subS) * paramsW[:, :80] 
        sub = subA + subB 

        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26):
        #for ind in range(6):
        for ind in range(4):
            inp_cat = torch.cat((inp[:, ind], sub), dim=1)
            out = self._nets_one[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088  --- 3072
        #print ("outs:", outs.shape)
        new_latent = latent.clone()
        #new_latent[:, :3072] = latent[:, :3072] + outs
        scale = 20.0 
        new_latent[:, :512] = latent[:, :512] + outs[:, :512] * scale 
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536] * scale 
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048] * scale 

        return new_latent

    def forwardA(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        #N = inp.shape[0]
        #latent = latent.view(N, 18, 512)
        paramsS = params[:, 144:224] - paramsW[:, 144:224]
        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26):
        #for ind in range(6):
        for ind in range(4):
            inp_cat = torch.cat((inp[:, ind], paramsS), dim=1)
            out = self._nets_one[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088  --- 3072
        #print ("outs:", outs.shape)
        new_latent = latent.clone()
        #new_latent[:, :3072] = latent[:, :3072] + outs
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048]

        return new_latent

    def forwardA3(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        #N = inp.shape[0]
        #latent = latent.view(N, 18, 512)
        paramsS = params[:, 144:224] - paramsW[:, 144:224]
        outs = []
        #for ind in range(26):
        #for ind in range(6):
        for ind in range(17): 
            inp_cat = torch.cat((inp[:, ind], paramsS), dim=1)
            out = self._nets_one[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088  --- 3072
        #print ("outs:", outs.shape)
        new_latent = latent.clone()
        #new_latent[:, :3072] = latent[:, :3072] + outs
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048]

        return new_latent   


    def forwardS(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        #N = inp.shape[0]
        #latent = latent.view(N, 18, 512)
        paramsS = params[:, :80] - paramsW[:, :80]
        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26):
        #for ind in range(6):
        for ind in range(4):
            inp_cat = torch.cat((inp[:, ind], paramsS), dim=1)
            out = self._nets_one[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088  --- 3072
        #print ("outs:", outs.shape)
        new_latent = latent.clone()
        #new_latent[:, :3072] = latent[:, :3072] + outs
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048]

        return new_latent
        """
        outs = outs + latent 
        return outs
        """

    def forwardA2(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        paramsE = (params[:, 144:224] - paramsW[:, 144:224])
        outs = []
        for ind in range(26):
            inp_cat = torch.cat((inp[:, ind], paramsE), dim=1)
            out = self._nets_one[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088
        outs = outs + latent
        return outs

    def forwardE(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        paramsE = params[:, 80:144] - paramsW[:, 80:144]
        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26): # entangle with light 
        for ind in range(9):
            inp_cat = torch.cat((inp[:, ind], paramsE), dim=1)
            out = self._nets_exp[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088
        new_latent = latent.clone()
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3584] = latent[:, 2560:3584] + outs[:, 1536:2560] 
        new_latent[:, 4096:5120] = latent[:, 4096:5120] + outs[:, 2560:3584] 
        new_latent[:, 5632:6656] = latent[:, 5632:6656] + outs[:, 3584:4608] 

        #return outs
        return new_latent 

        #outs = outs + latent
        #return outs

    def forwardP(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        paramsP = params[:, 224:227] - paramsW[:, 224:227]
        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26):
        for ind in range(4):
            inp_cat = torch.cat((inp[:, ind], paramsP), dim=1)
            out = self._nets_pose[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)# batch, 9088
        #outs = outs  + latent
        new_latent = latent.clone()
        #new_latent[:, :3072] = latent[:, :3072] + outs
        new_latent[:, :512] = latent[:, :512] + outs[:, :512]
        new_latent[:, 1024:2048] = latent[:, 1024:2048] + outs[:, 512:1536]
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 1536:2048]
        #return outs
        return new_latent 

    def forwardL(self, inp, params, latent, paramsW):
        """
        inp: batch, 26, 32
        params: batch, 80
        latent: batch, 9088
        """
        scale = 1.0
        #paramsL = (params[:, 227:254] - paramsW[:, 227:254]) * scale
        paramsL = params[:, 227:254]# - paramsW[:, 227:254]) * scale
        #paramsS = params[:, :80]
        outs = []
        #for ind in range(26):
        #for ind in range(7): 
        for ind in range(9):  
        #for ind in range(5):  
            inp_cat = torch.cat((inp[:, ind], paramsL), dim=1)
            out = self._nets_light[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            #out = self._nets[ind](inp_cat) # batch, 512, 256, 128, 64, 32
            out = self._nets[ind](out)  # batch, 512, 256, 128, 64, 32
            #out = out.unsqueeze(1)
            outs.append(out)
        #outs = torch.cat(outs, dim=1)# batch, 9088
        #save_outs = outs.detach().cpu().squeeze().numpy() * 10.0
        #import numpy
        #numpy.savetxt("testlight.csv", save_outs, delimiter=',')
        #outs = outs  + latent
        #return outs
        outs = torch.cat(outs, dim=1)# batch, 9088
        new_latent = latent.clone()
        """
        scale = 1.0
        new_latent[:, 1536:2048] = latent[:, 1536:2048] + outs[:, :512] * scale
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, 512:1024] * scale
        new_latent[:, 4096:5120] = latent[:, 4096:5120] + outs[:, 1024:2048] * scale
        new_latent[:, 5632:6656] = latent[:, 5632:6656] + outs[:, 2048:3072] * scale
        new_latent[:, 7168:7680] = latent[:, 7168:7680] + outs[:, 3072:3584] * scale
        """
        scale = 1.0 #for light right layers 
        new_latent[:, 2048:3072] = latent[:, 2048:3072] + outs[:, :1024] * scale
        new_latent[:, 3584:4608] = latent[:, 3584:4608] + outs[:, 1024:2048] * scale
        new_latent[:, 5120:7680] = latent[:, 5120:7680] + outs[:, 2048:] * scale
        # fix RGB 
        new_latent[:, 2048:2560] = latent[:, 2048:2560]
        new_latent[:, 3584:4096] = latent[:, 3584:4096]
        new_latent[:, 5120:5632] = latent[:, 5120:5632]
        new_latent[:, 6656:7168] = latent[:, 6656:7168]
        """
        scale = 1.0 #for light right layers 
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, :512] * scale
        new_latent[:, 3584:4096] = latent[:, 3584:4096] + outs[:, 512:1024] * scale
        new_latent[:, 5632:6656] = latent[:, 5632:6656] + outs[:, 1024:2048] * scale




        scale = 1.0 not work  
        new_latent[:, 2560:3072] = latent[:, 2560:3072] + outs[:, :512] * scale
        new_latent[:, 4096:4608] = latent[:, 4096:4608] + outs[:, 512:1024] * scale
        new_latent[:, 5632:6656] = latent[:, 5632:6656] + outs[:, 1024:2048] * scale
        new_latent[:, 7168:7680] = latent[:, 7168:7680] + outs[:, 2048:] * scale
        """

        return new_latent

