import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        
        self._parser.add_argument('--data_dir', type=str, default='sample_dataset', help='path to dataset')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--dataset_mode', type=str, default='StyleGAN2Dataset', help='chooses dataset to be used')
        self._parser.add_argument('--image_dir', type=str, default='Images', help='dir to load images')
        self._parser.add_argument('--params_path', type=str, default='params.pkl', help='params path')
        self._parser.add_argument('--landmarks_path', type=str, default='landmarks.pkl', help='landmarks path')
        self._parser.add_argument('--latents_path', type=str, default='latents.pkl', help='latents path')
        self._parser.add_argument('--constants_path', type=str, default='constants.pkl', help='constants path')
        self._parser.add_argument('--name', type=str, default='test',
                                  help='name of the experiment. It decides where to store samples and models')

        self._parser.add_argument('--apnet_name', type=str, default='apnet_wpdc', help='APModel name of the experiment')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--load_apnet_epoch', type=int, default=200, help='which APNet epoch to load?')
        self._parser.add_argument('--model', type=str, default='APModel', help='model to run: APModel | RIGModelS')

        self._parser.add_argument('--train_list', type=str, default='train_list.txt', help='training data')            
        self._parser.add_argument('--test_list', type=str, default='test_list.txt', help='testing data')
        
        self._parser.add_argument('--img_size', type=int, default=224, help='input image size | does not influence StyleGAN2')

        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--finetune', action='store_true', help='if true, finetune the model')
        
        self._parser.add_argument('--train_render', action='store_true', help='if True, using render loss to train the model')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        #self._opt = self._parser.parse_args()
        self._opt = self._parser.parse_known_args()[0]

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
