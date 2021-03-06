#!/usr/bin/env python3

import sys
import pickle
import math
import datetime as dt
import numpy as np

import torch.nn as nn
import torch.nn.init as init

from sympy import pi as PI
from datasets import DATA_SCALER

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def sidc_filter(data):
    newdata = data[:]
    for idx, val in enumerate(data[6:-6]):
        idx += 6
        newdata[idx] = (0.5*data[idx-6] + data[idx-5] + data[idx-4] + data[idx-3] + data[idx-2] + data[idx-1] + data[idx]+\
                    data[idx+1] + data[idx+2] + data[idx+3] + data[idx+4] + data[idx+5] + data[idx+6]*0.5)/12

    return newdata

def get_cycles(ssn_dataset):
    try:
        cycle_file = open("cycle_data.pickle", "rb")
        CYCLE_DATA = pickle.load(cycle_file)
        return CYCLE_DATA
    except:
        CYCLE_DATA = {"start_date":[], "end_date":[], "max_date":[], "solar_max":[], "length":[]}
    data = []
    for idx, year in enumerate(ssn_dataset.yeardata):
        for month, ssn in enumerate(ssn_dataset.valdata[idx]):
            data.append(float(ssn))

    print(data)
    data = sidc_filter(data)
    print(data)

    curr_min = [ssn_dataset.yeardata[0], 1, 500]
    curr_max = [ssn_dataset.yeardata[0], 1, 0]
    CYCLE_DATA["start_date"].append([ssn_dataset.yeardata[0], 1])

    for idx, ssn in enumerate(data):
        year = ssn_dataset.yeardata[0] + idx//12
        month = idx%12 + 1

        if ssn <= curr_min[2]:
            curr_min = [year, month, ssn]

        if ssn >= curr_max[2]:
            curr_max = [year, month, ssn]

        if (year > curr_min[0] + 5) or (len(data) - idx == 6):
            CYCLE_DATA["end_date"].append([curr_min[0], curr_min[1] - 1])
            CYCLE_DATA["length"].append((CYCLE_DATA["end_date"][-1][0]-\
            CYCLE_DATA["start_date"][-1][0])*12 + (CYCLE_DATA["end_date"][-1][1]-\
            CYCLE_DATA["start_date"][-1][1] + 1))

            CYCLE_DATA["start_date"].append([curr_min[0], curr_min[1]])
            curr_min[2] = 500

        if (year > curr_max[0] + 5 and len(CYCLE_DATA["max_date"]) < len(CYCLE_DATA["start_date"]))\
        or(len(data) - idx == 6):
            CYCLE_DATA["max_date"].append([curr_max[0], curr_max[1]])
            CYCLE_DATA["solar_max"].append(curr_max[2])
            curr_max[2] = 0

    CYCLE_DATA["end_date"].append([year + (month-1)//12, (month-1)%12])
    CYCLE_DATA["length"].append((CYCLE_DATA["end_date"][-1][0]-\
    CYCLE_DATA["start_date"][-1][0])*12 + (CYCLE_DATA["end_date"][-1][1]-\
    CYCLE_DATA["start_date"][-1][1] + 1))
    CYCLE_DATA["max_date"].append([curr_max[0], curr_max[1]])
    CYCLE_DATA["solar_max"].append(curr_max[2])

    cycle_file = open("cycle_data.pickle", "wb")
    pickle.dump(CYCLE_DATA, cycle_file)
    cycle_file.close()

    return CYCLE_DATA

def gen_samples(ssn_data, aa_data, cycle_data, cycle, normalize=False, tf=None):
    samples = []
    timestamps = []

    start_date = cycle_data["start_date"][cycle - 1]
    end_date = cycle_data["end_date"][cycle - 1]

    if tf==None:
        tf = (end_date[0] - start_date[0])*12 + (end_date[1] - start_date[1] + 1)

    for step in range(tf):
        month_num = (step % 12) + 1
        year_index  = (step // 12) + 1

        month = (start_date[1] + month_num - 1) % 12
        year = (start_date[0] + year_index - 1) + (start_date[1] + month_num - 1)//12

        delayed_ssn = ssn_data.data[year][month]
        delayed_aa = aa_data.data[year][month]

        ms = math.sin((2*PI*month_num)/12)
        mc = math.cos((2*PI*month_num)/12)
        ys = math.sin((2*PI*year_index)/11)
        yc = math.cos((2*PI*year_index)/11)

        month = (end_date[1] + month_num - 1) % 12
        year = (end_date[0] + year_index - 1) + (end_date[1] + month_num  - 1)//12

        samples.append(np.array([ys, yc, ms, mc, delayed_aa, delayed_ssn]))
        timestamps.append(dt.datetime(year=year, month=month+1, day=15))

    if normalize:
        samples = DATA_SCALER.transform(samples)

    return timestamps, samples

def print_cycles(cycle_data):
    print("{}{: >15}{: >15}{: >15}{: >15}{: >20}"\
    .format("SC Number","Start Date","End Date","Max Date","Solar Max","Length(in months)"))
    for idx in range(len(cycle_data["start_date"])):
        print("{: >10}{: >15}{: >15}{: >15}{: >15}{: >20}"\
        .format(idx, str(cycle_data["start_date"][idx]),\
        str(cycle_data["end_date"][idx]), str(cycle_data["max_date"][idx]),\
        round(cycle_data["solar_max"][idx],2), cycle_data["length"][idx]))

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
