from .model_factory import BaseModel
from networks.network_factory import NetworksFactory
from collections import OrderedDict
import torch
from loss.vdc_loss import VDCLoss
#from loss.render_loss import RenderLoss
from loss.pdc_loss import PDCLoss
from loss.wpdc_loss import WPDCLoss
#from loss.landmark_loss import WingLoss, LandmarkLoss
from utils.lr_scheduler import WarmupMultiStepLR
from bfm.bfm import BFM

class APModel(BaseModel):
    def __init__(self, opt, is_train):
        super(APModel, self).__init__(opt, is_train)
        self._name = 'APModel'

        # create networks
        self._init_create_networks()

        # use pre-trained DFRModel
        if self._is_train and not self._opt.load_epoch > 0:
            self._init_weights()

        # init rendered

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        if self._opt.finetune:
            self.load_finetune()

        if not self._is_train:
            self.set_eval()

        # init loss
        self._init_losses()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # init BFM basis
        # self.facemodel = BFM("bfm/BFM/mSEmTFK68etc.chj")

    def _init_create_networks(self):
        network_type = 'APNet'
        self.network = self._create_branch(network_type)
        if len(self._gpu_ids) > 1:
            self.network = torch.nn.DataParallel(self.network, device_ids=self._gpu_ids)
        if torch.cuda.is_available():
            self.network.cuda()

    def _init_train_vars(self):
        self._current_lr = self._opt.learning_rate
        self._decay_rate = self._opt.decay_rate
        # initialize optimizers
        #self._optimizer = torch.optim.SGD([
        #                {'params': self.network.parameters()},
        #                #{'params': self.renderer.parameters()}
        #                ], lr=self._current_lr, momentum=self._decay_rate)
        self._optimizer = torch.optim.Adam([
            {'params': self.network.parameters()},
            # {'params': self.renderer.parameters()}
        ], lr=self._current_lr, betas=(0.9, 0.99))
        self._scheduler = WarmupMultiStepLR(
            self._optimizer,
            #[100, 200],
            [20, 40, 60, 80],
            gamma = 0.01,
            warmup_epochs = 5,
        )

    def _init_weights(self):
        pass

    def load(self):
        load_epoch = self._opt.load_epoch
        self._load_network(self.network, 'APNet', load_epoch, self._opt.name)
        #self._load_network(self.renderer, 'Render', load_epoch, self._opt.name)

    # def load_finetune(self):
    #     load_epoch = self._opt.load_finetune_epoch
    #     self._load_network(self.network, 'DFRNet', load_epoch, self._opt.finetune_name)
    #     #self._load_network(self.renderer, 'Render', load_epoch, self._opt.finetune_name)
    

    def save(self, label):
        # save networks
        self._save_network(self.network, 'APNet', label, self._opt.name)
        #self._save_network(self.renderer, 'Render', label, self._opt.name)


    def _create_branch(self, branch_name):
        return NetworksFactory.get_by_name(branch_name)

    def _init_losses(self):
        # define loss function
        #self._MSE_loss = torch.nn.MSELoss(reduction='mean')
        #self._SML1_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self._VDC_loss = VDCLoss()
        self._WPDC_loss = WPDCLoss()
        self._PDC_loss = PDCLoss()
        #self._LANDMARK_loss = LandmarkLoss() #WingLoss()
        #self._RENDER_loss = RenderLoss(self._opt)

        if torch.cuda.is_available():
            #self._MSE_loss = self._MSE_loss.cuda()
            self._VDC_loss = self._VDC_loss.cuda()
            self._WPDC_loss = self._WPDC_loss.cuda()
            self._PDC_loss = self._PDC_loss.cuda()
            #self._LANDMARK_loss = self._LANDMARK_loss.cuda()
            #self._SML1_loss = self._SML1_loss.cuda()
            #self._RENDER_loss = self._RENDER_loss.cuda()

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
        params = train_batch['param'].float()
        if torch.cuda.is_available():
            latent = latent.cuda()
            params = params.cuda()
        param_lst = self.network(latent)

        loss = 0
        if self._opt.train_vdc:
            self.vdc_loss = self._opt.weight_vdc * self._VDC_loss(param_lst, params) 
            loss += self.vdc_loss 
        if self._opt.train_pdc:
            self.pdc_loss = self._opt.weight_pdc * self._PDC_loss(param_lst, params, epoch)
            loss += self.pdc_loss 
        if self._opt.train_wpdc:
            self.wpdc_loss = self._opt.weight_wpdc * self._WPDC_loss(param_lst, params) 
            loss += self.wpdc_loss 
        # if self._opt.train_render:
        #     images = train_batch['inp_img'].float()
        #     images = images.cuda()
        #     self.render_loss = self._opt.weight_render * self._RENDER_loss(param_lst, params, images)
        #     loss += self.render_loss
        # if self._opt.train_landmark:
        #     self.landmark_loss = self._opt.weight_landmark * self._LANDMARK_loss(param_lst, params)
        #     loss += self.landmark_loss
        
        return loss 


    def forward_test(self, latent):
        self.set_eval()
        with torch.no_grad():
            latent = self._FloatTensor(latent)
            param_lst = self.network(latent)  #  (1, 257)
            #alpha_shp, alpha_exp, albedo, angles, gamma, translation = parse_param(param_lst)
        return param_lst

    def get_current_errors(self):
        loss_dict = OrderedDict()
        if self._opt.train_vdc:
            loss_dict['loss_vdc'] = self.vdc_loss.data
        if self._opt.train_pdc:
            loss_dict['loss_pdc'] = self.pdc_loss.data 
        if self._opt.train_wpdc:
            loss_dict['loss_wpdc'] = self.wpdc_loss.data 
        # if self._opt.train_render:
        #     loss_dict['loss_render'] = self.render_loss.data
        # if self._opt.train_landmark:
        #     loss_dict['loss_landmark'] = self.landmark_loss.data
        
        return loss_dict

    def set_train(self):
        self.network.train()
        self._is_train = True

    def set_eval(self):
        self.network.eval()
        self._is_train = False

