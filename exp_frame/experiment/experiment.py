import logging
import multiprocessing.context
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import reduce, wraps
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace
from typing import List

from util.database_utils import auto_insert_database
from util.other_utils import dict_to_str, send_email, mkdir, now_time


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        func_name = func.__name__

        begin = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()

        run_time = end - begin
        self.__dict__[f'{func_name}_time'] = run_time
        return result

    return wrapper


def skippable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        func_name = func.__name__
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f'{func_name} error: {e}')
            self._exceptions.append((func_name, e))
        return result

    return wrapper


def limiter(timeout):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            func_name = func.__name__
            result = None

            pool = ThreadPool(processes=1)
            future = pool.apply_async(func, args=args, kwds=kwargs)
            try:
                result = future.get(timeout)
            except multiprocessing.context.TimeoutError as e:
                print(f'{func_name} timeout: {timeout}s')
                self._exceptions.append((func_name, e))
                pool.terminate()
            return result

        return wrapper

    return decorator


def asyncable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(func, *args, **kwargs)

    return wrapper


class Experiment(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self._name = None
        self._model = None
        self._data = SimpleNamespace()
        self._exceptions = []

        mkdir(self.model_path)
        mkdir(self.log_path)

        _, date = now_time()
        self.model_path = os.path.join(self.model_path, f'{self.name}_{date}')
        self.log_path = os.path.join(self.log_path, f'{self.name}_{date}.log')

        self._create_logger()

    def _create_logger(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        logger_handler = logging.FileHandler(self.log_path)
        logger_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        logger_handler.setFormatter(formatter)
        for h in self._logger.handlers:
            self._logger.removeHandler(h)
        self._logger.addHandler(logger_handler)

    def data_load(self):
        raise NotImplementedError

    def model_build(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def model_save(self):
        raise NotImplementedError

    def model_load(self):
        raise NotImplementedError

    @skippable
    def dataset_update(self, database_config, name=None, exp_data=None):
        if name is None:
            name = self.name.lower()
        if exp_data is None:
            exp_data = self.exp_data
        auto_insert_database(database_config, exp_data, name)

    @limiter(3)
    def email(self, email_config, receiver, title=None, text=None):
        if title is None:
            title = f'{self.name} Experiment Finished'
        if text is None:
            text = self.report()
        send_email(receiver=receiver,
                   title=title,
                   text=text,
                   **email_config)

    def report(self):
        exp_data = self.exp_data
        sub_time = list(filter(lambda k: k[-5:] == '_time', exp_data.keys()))
        whole_time = reduce(lambda x, y: x + y, map(lambda k: exp_data[k], sub_time))

        report_lines = [f'whole time: {whole_time}']
        for time in sub_time:
            time_str = time.replace('_', ' ')
            report_lines.append(f'{time_str}: {exp_data.pop(time)}')
        report_lines.append('=' * 20)

        report_lines.append(dict_to_str(exp_data))

        report = '\n'.join(report_lines)
        return report

    def __str__(self):
        return dict_to_str(self.exp_data)

    @property
    def exp_data(self, ignore: List[str] = None):
        exp_data = self.__dict__.copy()
        private = list(filter(lambda k: k[0] == '_', exp_data.keys()))
        for p in private:
            exp_data.pop(p)
        if ignore is not None:
            for i in ignore:
                exp_data.pop(i)
        return exp_data

    @property
    def name(self):
        if self._name is not None:
            return self._name
        classname = type(self).__name__
        lower_classname = classname.lower()
        exp_loc = lower_classname.rfind('exp')
        exp_name = classname[0:exp_loc]
        return exp_name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def exceptions(self):
        return self._exceptions

    def log_info(self):
        self._logger.info(self.report())
