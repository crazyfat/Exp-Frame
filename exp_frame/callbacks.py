from copy import deepcopy
from typing import Dict, Any

import numpy as np


class EarlyStopRecord:
    def __init__(self,
                 min_delta=0.0,
                 patience=0,
                 mode_max=True):
        self.patience = patience
        self.mode_max = mode_max
        self.min_delta = min_delta
        self.best_record = None
        self.wait = 0
        self.stopped = False

        if self.mode_max:
            self.compare = lambda v: v - self.best_record > self.min_delta
            self.best_record = -np.inf
        else:
            self.compare = lambda v: self.best_record - v > self.min_delta
            self.best_record = np.inf

    def init(self):
        self.stopped = False
        self.wait = 0
        if self.mode_max:
            self.best_record = -np.inf
        else:
            self.best_record = np.inf

    def update(self, value):
        if self.compare(value):
            self.best_record = value
            self.wait = 0
            self.stopped = False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True

    def __bool__(self):
        return self.stopped


class EarlyStop:
    def __init__(self,
                 model,
                 min_delta=0.0,
                 patience=0,
                 mode_max=True,
                 verbose=1):
        self._model = model
        self.verbose = verbose
        self.mode_max = mode_max
        self.patience = patience
        self.min_delta = min_delta
        self.records = None
        self.best_weights = None
        self.stopped_epoch = 0
        self.best_epoch = 0

    def update(self, epoch: int, results: Dict[str, Any]):
        if self.records is None:
            self.records = {name: EarlyStopRecord(patience=self.patience,
                                                  mode_max=self.mode_max,
                                                  min_delta=self.min_delta) for name in results.keys()}
        if self.verbose > 0:
            for name, record in self.records.items():
                print(f'last {name}: {record.best_record}')

        for name in results.keys():
            self.records[name].update(results[name])

        if self.verbose > 0:
            for name, record in self.records.items():
                print(f'{name} wait: {record.wait}')

        if all(self.records.values()):
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.stopped_epoch = epoch
            self._model.load_state_dict(self.best_weights)
            return True
        else:
            update_records_count = len(list(filter(lambda r: r.wait == 0, self.records.values())))
            if update_records_count > 0:
                self.best_weights = deepcopy(self._model.state_dict())
                self.best_epoch = epoch
                if self.verbose > 0:
                    print(f'record best epoch: {epoch}')
