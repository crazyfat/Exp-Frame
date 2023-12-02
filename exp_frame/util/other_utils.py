import os

import numpy as np


def function_timer(fun):
    from datetime import datetime

    def wrapper(*args, **kwargs):
        begin = datetime.now()
        result = fun(*args, **kwargs)
        end = datetime.now()
        return result, end - begin

    return wrapper


def print_array_python(array: np.ndarray, deep=0):
    deep += 1
    if len(array.shape) == 1:
        return '[{}]'.format(', '.join([str(i) for i in array]))
    else:
        return '[{}]'.format(',\n{}'.format(' ' * deep).join([print_array_python(i, deep) for i in array]))
    pass


def static(mat):
    max_v = np.max(mat)
    min_v = np.min(mat)
    print('max:', max_v)
    print('min:', min_v)

    print('mean:', np.mean(mat))
    print('median:', np.median(mat))
    print('std:', np.std(mat))

    print('histogram:')
    f, g = np.histogram(mat)
    print(f)
    print(g)

    print('percentile:')
    percents = np.arange(99, 100, 0.1)
    print(percents)
    print(np.percentile(mat, percents))

    percents = np.arange(99.9, 100, 0.01)
    print(percents)
    print(np.percentile(mat, percents))


def now_time():
    from datetime import datetime
    now = datetime.now()
    return now, now.strftime('%Y-%m-%d_%H-%M-%S')


def send_email(receiver, title, text, smtp_server=None, mail_user=None, mail_pass=None, **kwargs):
    import smtplib
    import ssl
    from email.header import Header
    from email.mime.text import MIMEText

    if 'smtp_server' in kwargs:
        smtp_server = kwargs['smtp_server']
    if 'mail_host' in kwargs:
        smtp_server = kwargs['mail_host']
    if 'mail_user' in kwargs:
        mail_user = kwargs['mail_user']
    if 'mail_pass' in kwargs:
        mail_pass = kwargs['mail_pass']

    missing_params = []
    if mail_pass is None:
        missing_params.append('mail_pass')
    if mail_user is None:
        missing_params.append('mail_user')
    if smtp_server is None:
        missing_params.append('smtp_server')
    if len(missing_params) != 0:
        missing = ', '.join(missing_params)
        print(f'cannot find {missing}')
        return
    # 第三方 SMTP 服务
    sender = mail_user

    message = MIMEText(text, 'plain', 'utf-8')
    subject = title
    message['Subject'] = Header(subject, 'utf-8')
    try:
        context = ssl.create_default_context()
        port = 465
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(mail_user, mail_pass)
            server.sendmail(sender, receiver, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")


def dict_to_str(d: dict, split='\n'):
    return split.join(map(lambda i: '{} : {}'.format(i[0], i[1]), d.items()))


def mkdir(path):
    from os import makedirs
    from os.path import exists
    if exists(path) is not True:
        makedirs(path)


def visit_dict(d: dict, func):
    return {k: visit_dict(v, func) if isinstance(v, dict) else func(v) for k, v in d.items()}


def serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()

    return obj


def print_ragged_tensor(ragged_tensor):
    values = ragged_tensor.values.numpy()
    row_splits = ragged_tensor.row_splits.numpy()

    part = []
    start = 0
    for split in row_splits[1:]:
        part.append(values[start:split])
        start = split

    s = ',\n'.join(map(str, part))
    lines = [' ' + line for line in s.splitlines()]
    s = '\n'.join(lines)
    print(f'[{s[1:]}]')


def list_folders(path):
    return filter(lambda sub: os.path.isdir(os.path.join(path, sub)), os.listdir(path))


def list_files(path):
    return filter(lambda sub: os.path.isfile(os.path.join(path, sub)), os.listdir(path))
