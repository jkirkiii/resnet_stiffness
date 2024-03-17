import csv
import datetime
import pathlib
import random

def generate_color_transition(start_color, end_color, length):
    if length == 1: return [start_color]
    # convert hex colors to RGB tuples
    start_rgb = list(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = list(int(end_color[i:i+2], 16) for i in (1, 3, 5))

    # calculate the step size for each RGB component
    r , g , b  = start_rgb
    r_, g_, b_ = end_rgb
    dr = (r_ - r) / (length-1)
    dg = (g_ - g) / (length-1)
    db = (b_ - b) / (length-1)

    color_list = []

    for _ in range(length):
        color_list += [f'#{int(r):02x}{int(g):02x}{int(b):02x}']
        r += dr
        g += dg
        b += db

    return color_list

def generate_random_colors(length): return [f'#{random.randint(0, 0xffffff):06x}' for _ in range(length)] # generate a random hex color code

def logger(row, path: pathlib.Path, *, name='log', verbose=True):
    log_path = path / f'{name}.csv'
    with open(log_path, 'a') as file: csv.writer(file).writerow(row)
    if verbose: print(*row)
    return log_path

def format_time(seconds, fine=False):
    time_string = str(datetime.timedelta(seconds=seconds))
    return time_string if fine else time_string.split('.')[0]
