from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
       
        self._parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self._parser.add_argument('--serial_batches', action='store_true', default=True,
                                  help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--n_threads_test', default=0, type=int, help='# threads for loading data')
        self._parser.add_argument('--save_dir', type=str, default='results', help='path to dataset')
        self.is_train = False
