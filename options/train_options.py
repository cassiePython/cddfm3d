from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # TODO: delete non-used arguments as many losses are not used in the final version.
        self._parser.add_argument('--serial_batches', action='store_true',
                                  help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--n_threads_train', default=0, type=int, help='# threads for loading data')
        self._parser.add_argument('--total_epoch', type=int, default=100, help='total epoch for training')
        self._parser.add_argument('--save_interval', type=int, default=20, help='interval for saving models')
        self._parser.add_argument('--show_interval', type=int, default=10, help='interval for printing loss')
        self._parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
        self._parser.add_argument('--weight_landmark', type=float, default=1.0, help='landmark training weight')
        self._parser.add_argument('--weight_render', type=float, default=1.0, help='render training weight')
        self._parser.add_argument('--weight_pdc', type=float, default=1.0, help='PDC training weight')
        self._parser.add_argument('--weight_vdc', type=float, default=1.0, help='VDC training weight')
        self._parser.add_argument('--weight_wpdc', type=float, default=1.0, help='WPDC training weight')
        self._parser.add_argument('--weight_rec', type=float, default=1.0, help='REC training weight')
        self._parser.add_argument('--weight_image', type=float, default=1.0, help='IMAGE training weight')
        self._parser.add_argument('--weight_edge', type=float, default=1.0, help='Edge training weight')
        self._parser.add_argument('--weight_bg', type=float, default=1.0, help='BG training weight')
        self._parser.add_argument('--weight_g', type=float, default=1.0, help='Gnerator training weight')
        self._parser.add_argument('--weight_mse', type=float, default=1.0, help='Reconstruct training weight')
        self._parser.add_argument('--weight_adv', type=float, default=1.0, help='Adversarial training weight')
        self._parser.add_argument('--weight_percept', type=float, default=1.0, help='Generator training weight')
        self._parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate')
        self._parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        self._parser.add_argument('--train_pdc', action='store_true', help='if True, using pdc loss to train the model')
        self._parser.add_argument('--train_wpdc', action='store_true', help='if True, using wpdc loss to train the model')
        self._parser.add_argument('--train_vdc', action='store_true', help='if True, using vdc loss to train the model')
        self._parser.add_argument('--train_render', action='store_true', help='if True, using render loss to train the model')
        self._parser.add_argument('--train_landmark', action='store_true', help='if True, using landmark loss to train the model')
        self._parser.add_argument('--train_rec', action='store_true', help='if True, using rec loss to train the model')
        self._parser.add_argument('--train_image', action='store_true', help='if True, using image loss to train the model')
        self._parser.add_argument('--train_edge', action='store_true', help='if True, using edge loss to train the model')
        self._parser.add_argument('--train_bg', action='store_true', help='if True, using bg loss to train the model')
        self._parser.add_argument('--train_g', action='store_true', help='if True, using generator loss to train the model')
        self._parser.add_argument('--train_mse', action='store_true', help='if True, using mse loss to train the model')
        self._parser.add_argument('--train_adv', action='store_true', help='if True, using adv loss to train the model')

        self.is_train = True
