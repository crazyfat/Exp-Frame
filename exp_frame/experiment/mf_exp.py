from os.path import join
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks import EarlyStop
from dataset.datasets import RatingDataset
from experiment import Experiment, timer
from metics import Mean
from pytorch_models.MF import MF
from sensitive_info import database_config, email_config
from util.exp_utils import load_obj, xavier_init
from util.other_utils import dict_to_str, function_timer


class MatrixFactorizationExp(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(dict_to_str(self.exp_data))
        self._loss_tracker = Mean(name='loss')

    @timer
    def data_load(self):
        num_workers = 0
        self._data = SimpleNamespace()

        # dataset number
        num_users, num_items, num_words = load_obj(join(self.data_dir, 'num_users_items'))
        self.total_users = num_users
        self.total_items = num_items

        # train dataset
        train_user_reviews_path = join(self.data_dir, 'train_user_reviews')
        test_s_reviews = load_obj(join(self.data_dir, 'test_s_reviews'))
        val_s_reviews = load_obj(join(self.data_dir, 'val_s_reviews'))

        user_reviews = load_obj(train_user_reviews_path)
        train_dataset = RatingDataset(user_reviews, test_s_reviews, val_s_reviews)
        self._data.train = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers,
                                      pin_memory=True)

        # test dataset
        test_reviews = load_obj(join(self.data_dir, 'test_reviews'))
        val_reviews = load_obj(join(self.data_dir, 'val_reviews'))
        test_dataset = RatingDataset(test_reviews)
        val_dataset = RatingDataset(val_reviews)
        self._data.test = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=num_workers,
                                     pin_memory=True)
        self._data.val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers,
                                    pin_memory=True)

    @timer
    def model_build(self):
        model = MF(self.total_users, self.total_items, self.latent_size)
        if torch.cuda.is_available():
            model = model.cuda()
        xavier_init(model, [])
        self._model = model

    def batch_train(self, optimizer, criterion, epoch):
        self._model.train()
        self._loss_tracker.reset_state()
        progress = tqdm(self._data.train)
        progress.set_description_str(f'epoch: {epoch}/{self.epochs}')
        progress.set_postfix({self._loss_tracker.name: self._loss_tracker.result()})
        for data in progress:
            data = map(lambda d: d.cuda(), data)
            (users, items, true_rating) = data

            pred_rating = self._model(users, items)
            loss = criterion(pred_rating, true_rating)

            mean_loss = loss.mean()
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()

            self._loss_tracker.update_state(loss.cpu().detach().numpy().flatten())
            progress.set_postfix({self._loss_tracker.name: self._loss_tracker.result()})
        return self._loss_tracker.result()

    def evaluate_ranking(self, data):
        mae = nn.L1Loss(reduction='none')
        mse = nn.MSELoss(reduction='none')

        self._model.eval()
        with torch.no_grad():
            all_mae = []
            all_mse = []
            for data in tqdm(data, desc='eval'):
                data = map(lambda d: d.cuda(), data)
                (users, items, true_rating) = data

                pred_rating = self._model(users, items)
                all_mae.append(mae(pred_rating, true_rating).flatten().cpu())
                all_mse.append(mse(pred_rating, true_rating).flatten().cpu())

            eval_mae = torch.cat(all_mae, dim=0).mean()
            eval_mse = torch.cat(all_mse, dim=0).mean()

        return {'mae': eval_mae.numpy(), 'mse': eval_mse.numpy()}

    @timer
    def train(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='none')

        early_stop = None
        if self.early_stop_patience >= 0:
            early_stop = EarlyStop(model=self._model,
                                   patience=self.early_stop_patience,
                                   mode_max=False)

        for epoch in range(1, self.epochs + 1):
            batch_train = function_timer(self.batch_train)
            loss, batch_time = batch_train(optimizer, criterion, epoch)
            result = self.evaluate_ranking(self._data.val)

            log = {'epoch': epoch, 'loss': loss, 'batch_time': batch_time}
            log.update(result)

            self._logger.info(dict_to_str(log, split=' | '))

            print(dict_to_str(result, ', '))
            result.pop('mae')

            if early_stop is not None and early_stop.update(epoch, result):
                break
        if early_stop is not None:
            self.best_epoch = early_stop.best_epoch
            self.stopped_epoch = early_stop.stopped_epoch

    @timer
    def evaluate(self):
        result = self.evaluate_ranking(self._data.test)
        print(dict_to_str(result))
        self.__dict__.update(result)

    @timer
    def model_save(self):
        torch.save(self._model.state_dict(), self.model_path)

    @timer
    def model_load(self):
        model = MF(self.total_users, self.total_items, self.latent_size)
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load(self.model_path))
        self._model = model


def main(hyper_params):
    dataset = hyper_params['dataset']
    support = hyper_params['n_support']
    fold = hyper_params['fold']
    hyper_params['data_dir'] = f'./few_shot_data/{dataset}_s_{support}/5_core/{fold}_fold'

    exp = MatrixFactorizationExp(**hyper_params)
    exp.data_load()
    exp.model_build()
    exp.train()
    exp.model_save()
    exp.evaluate()
    exp.log_info()
    exp.dataset_update(database_config)
    exp.email(receiver='haoran.x@outlook.com',
              email_config=email_config)
