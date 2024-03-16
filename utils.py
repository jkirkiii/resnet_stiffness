import csv
import datetime
import pathlib

def logger(row, path: pathlib.Path, *, name='log', verbose=True):
    log_path = path / f'{name}.csv'
    with open(log_path, 'a') as file: csv.writer(file).writerow(row)
    if verbose: print(*row)
    return log_path

def format_time(seconds, fine=False):
    time_string = str(datetime.timedelta(seconds=seconds))
    return time_string if fine else time_string.split('.')[0]
