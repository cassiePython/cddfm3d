from __future__ import division
from models.model_factory import ModelsFactory
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()
        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        self._dataset_train = data_loader_train.load_data()
        self._dataset_train_size = len(data_loader_train)
        print('#train images = %d' % self._dataset_train_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt, is_train=True)

        self._train()

    def _train(self):
        self._steps_per_epoch = int (self._dataset_train_size / self._opt.batch_size)
        
        for i_epoch in range(self._opt.load_epoch + 1, self._opt.total_epoch + 1):
            # train epoch
            self._train_epoch(i_epoch)
            self._model.update_learning_rate()

            # save model
            if i_epoch % self._opt.save_interval == 0:
                print('saving the model at the end of epoch %d' % i_epoch)
                self._model.save(i_epoch)

    def _train_epoch(self, i_epoch):

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            
            self._model.optimize_parameters(train_batch, i_epoch)

            if i_train_batch % self._opt.show_interval == 0:
                self._display_terminal_train(i_epoch, i_train_batch)

    def _display_terminal_train(self, i_epoch, i_train_batch):
        errors = self._model.get_current_errors()
        message = '(epoch: %d, it: %d/%d) ' % (i_epoch, i_train_batch, self._steps_per_epoch)
        for k, v in errors.items():
            message += '%s:%f ' % (k, v)

        print(message)

if __name__ == "__main__":
    Train() 

