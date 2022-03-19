from .model_factory import BaseModel
from networks.network_factory import NetworksFactory
from collections import OrderedDict
import torch
from loss.render_loss import RenderLoss
from loss.shape_loss import ShapeLoss
from loss.landmark_loss import LandmarkLoss
from utils.lr_scheduler import WarmupMultiStepLR
from bfm.bfm import BFM


class RIGModel(BaseModel):
    def __init__(self, opt, is_train):
        super(RIGModel, self).__init__(opt, is_train)
        self._name = 'RIGModelS'

        # create networks
        self._init_create_networks()

        # use pre-trained RIGModel
        if self._is_train and not self._opt.load_epoch > 0:
            self._init_weights()

        # init pre-trained APModel
        self._init_APModel()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        if not self._is_train:
            self.set_eval()

        # init loss
        if self._is_train:
            self._init_losses()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # init BFM basis
        self.facemodel = BFM("bfm/BFM/mSEmTFK68etc.chj")

    def _init_create_networks(self):
        self.RigNetEncoder = self._create_branch("RigNetEncoder")
        self.RigNetDecoder = self._create_branch("RigNetDecoder")
        # Not fully tested on Multi-GPUs
        if len(self._gpu_ids) > 1:
            self.RigNetEncoder = torch.nn.DataParallel(self.RigNetEncoder, device_ids=self._gpu_ids)
            self.RigNetDecoder = torch.nn.DataParallel(self.RigNetDecoder, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.RigNetEncoder.cuda()
            self.RigNetDecoder.cuda() 

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        self._optimizer = torch.optim.SGD([
                        {'params': self.RigNetEncoder.parameters()},
                        {'params': self.RigNetDecoder.parameters()}
                        ], lr=self._current_lr, momentum=self._decay_rate)
        self._scheduler = WarmupMultiStepLR(
            self._optimizer,
            #[100, 200],
            [20, 40, 60, 80],
            gamma = 0.01,
            warmup_epochs = 5,
        )

    def _init_weights(self):
        # TODO: add initialization to improve performance
        pass

    def _init_APModel(self):
        self.APNet = self._create_branch("APNet")
        load_epoch = self._opt.load_apnet_epoch
        self._load_network(self.APNet, 'APNet', load_epoch, self._opt.apnet_name)
        if torch.cuda.is_available():
            self.APNet.cuda()

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(self.RigNetEncoder, 'RigNetEncoder', load_epoch, self._opt.name)
        self._load_network(self.RigNetDecoder, 'RigNetDecoder', load_epoch, self._opt.name)

    def save(self, label): 
        self._save_network(self.RigNetEncoder, 'RigNetEncoder', label, self._opt.name)
        self._save_network(self.RigNetDecoder, 'RigNetDecoder', label, self._opt.name)

    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name)

    def _init_losses(self):
        # define loss function
        self._MSE_loss = torch.nn.MSELoss(reduction='mean')
        self._LANDMARK_loss = LandmarkLoss()
        self._RENDER_loss = RenderLoss(self._opt)
        self._SHAPE_loss = ShapeLoss()

        if torch.cuda.is_available():
            self._MSE_loss = self._MSE_loss.cuda()
            self._LANDMARK_loss = self._LANDMARK_loss.cuda()
            self._RENDER_loss = self._RENDER_loss.cuda()
            self._SHAPE_loss = self._SHAPE_loss.cuda()


    def optimize_parameters(self, train_batch, epoch):
        if self._is_train:
            loss = self.forward(train_batch, epoch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def update_learning_rate(self):
        self._scheduler.step()


    def forward(self, train_batch, epoch):
        latent = train_batch['latent'].float()
        images = train_batch['inp_img'].float()
        if torch.cuda.is_available():
            latent = latent.cuda()
            images = images.cuda()
        
        N = latent.shape[0]

        #print ("---latent----")
        #print (latent)

        # Reconstruction 
        #I = self.RigNetEncoder(latent) 
        Pv = self.APNet(latent)
        #print ("XXX_PV__XXXXX")
        #print (Pv)
        I = self.RigNetEncoder(latent) 


        What = self.RigNetDecoder.forwardShape(I, Pv, latent, Pv)
        What_flat = What.view(N, -1)

        loss = 0
        if self._opt.train_rec:
            self.rec_loss = self._opt.weight_rec * self._MSE_loss(What_flat, latent) 
            loss += self.rec_loss 

        # Cycle-Consistent Per-Pixel Editing 
        SP = N // 2
        w_ce = latent[: SP] # (Batch-size/2, ...)
        v_ce = latent[SP:] # (Btach-size/2, ...)

        #print ("=======")
        #print (v_ce)
        #print (w_ce)

        

        Pv_ce = self.APNet(v_ce.view(SP, -1))
        #print ("XXX_PVCE_XXXXX")
        #print (Pv_ce)

        I_ce = self.RigNetEncoder(w_ce) 
        Pw_cc = self.APNet(w_ce.view(SP, -1))
        #print ("XXXX_PWCC_XXXX")
        #print (Pw_cc)
        What_ce = self.RigNetDecoder.forwardShape(I_ce, Pv_ce, w_ce, Pw_cc)
        Phat_ce = self.APNet(What_ce.view(SP, -1))
        #print ("XXXX_Phatce_XXXX")
        #print (Phat_ce)
        Pedit = Pv_ce.clone()
        Pedit[:, :80] = Phat_ce[:, :80]  
        Iv = images[SP:] 

        # Cycle-Consistent Per-pixel Consistency Loss
        Pconsist = Pw_cc.clone()
        Pconsist[:, 80:] = Phat_ce[:, 80:]
        Iw = images[: SP]

        #print ("XXXXXXXX")
        #print (Pv_ce)
        #print (Phat_ce)
        #print (Pw_cc)

        if self._opt.train_render:
            #print ("Pedit:", Pedit.shape)
            #print ("v_ce:", v_ce.shape) 
            self.render_loss_ce = self._opt.weight_render * self._RENDER_loss(Pedit, Pv_ce, Iv)
            self.render_loss_cc = self._opt.weight_render * self._RENDER_loss(Pconsist, Pw_cc, Iw)

            loss += self.render_loss_ce + self.render_loss_cc 

        if self._opt.train_landmark:
            self.landmark_loss_ce = self._opt.weight_landmark * self._LANDMARK_loss(Pedit, Pv_ce) 
            self.landmark_loss_cc = self._opt.weight_landmark * self._LANDMARK_loss(Pconsist, Pw_cc) 
            loss += self.landmark_loss_ce + self.landmark_loss_cc 

        if self._opt.train_edge:
            self.edge_loss_ce = self._opt.weight_edge * self._SHAPE_loss(Pv_ce, Phat_ce, Pw_cc)   
            loss += self.edge_loss_ce 

        return loss


    def forward_test(self, latent_w, pca_id, pca_val):
        self.set_eval()
        with torch.no_grad():
            latent_w = self._FloatTensor(latent_w).cuda()  # (1, 9088)
            #latent_w = self._FloatTensor(latent_w).cuda()  # (1, 9088)
            I = self.RigNetEncoder(latent_w)
            Pw = self.APNet(latent_w)
            Pv = Pw.clone()
            Pv[:, pca_id] = Pv[:, pca_id]+ pca_val
            What = self.RigNetDecoder.forwardShape(I, Pv, latent_w, Pw) 
            WhatF = What.view(1, -1)
            Pwhat = self.APNet(WhatF)
        return What, Pw, Pwhat

    def get_current_errors(self):
        loss_dict = OrderedDict()
        if self._opt.train_rec:
            loss_dict['loss_rec'] = self.rec_loss.data
        if self._opt.train_render:
            loss_dict['loss_render_ce'] = self.render_loss_ce.data 
            loss_dict['loss_render_cc'] = self.render_loss_cc.data 
        if self._opt.train_landmark:
            loss_dict['loss_landmark_ce'] = self.landmark_loss_ce.data 
            loss_dict['loss_landmark_cc'] = self.landmark_loss_cc.data
        if self._opt.train_edge:
            loss_dict['loss_edge_ce'] = self.edge_loss_ce.data
        
        return loss_dict

    def set_train(self):
        self.RigNetEncoder.train()
        self.RigNetDecoder.train()
        self._is_train = True

    def set_eval(self):
        self.RigNetEncoder.eval()
        self.RigNetDecoder.eval()
        self.APNet.eval()
        self._is_train = False  
