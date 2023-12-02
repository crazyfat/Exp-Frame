import json
import pickle

import torch


def list_split(seq, nums):
    seq = list(seq)
    nums = list(nums)
    out = []
    last = 0
    for num in nums:
        out.append(seq[last:num])
        last += num

    return out


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 2)


def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_obj_json(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)


def load_user_item_counts(hyper_params):
    user_count = load_obj(hyper_params['data_dir'] + 'user_count')
    item_count = load_obj(hyper_params['data_dir'] + 'item_count')
    return user_count, item_count


def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    f = open(log_file, 'a')
    f.write(s + '\n')
    f.close()


def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()


def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')


def log_end_epoch(hyper_params, metrics, epoch, time_elpased, metrics_on='(VAL)'):
    string2 = ""
    for m in metrics:
        string2 += " | " + m + ' = ' + str(metrics[m])
    string2 += ' ' + metrics_on

    ss = '-' * 89
    ss += '\n| end of epoch {} | time: {:5.2f}s'.format(epoch, time_elpased)
    ss += string2
    ss += '\n'
    ss += '-' * 89
    file_write(hyper_params['log_file'], ss)


def xavier_init(model, exclude=None):
    if exclude is None:
        exclude = []
    exclude = set(map(id, exclude))
    for p in model.parameters():
        if p.dim() > 1 and id(p) not in exclude:
            torch.nn.init.xavier_uniform_(p)
