#!/usr/bin/env python3

import sys
import pickle

import torch.nn as nn
import torch.nn.init as init

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

    data = sidc_filter(data)

    curr_min = [ssn_dataset.yeardata[0], 1, 500]
    curr_max = [ssn_dataset.yeardata[0], 1, 0]
    CYCLE_DATA["start_date"].append([ssn_dataset.yeardata[0], 1])

    for idx, ssn in enumerate(data):
        year = ssn_dataset.yeardata[0] + idx//12
        month = idx%12 + 1
        if year > curr_min[0] + 5:
            CYCLE_DATA["end_date"].append([curr_min[0], curr_min[1] - 1])
            CYCLE_DATA["length"].append((CYCLE_DATA["end_date"][-1][0]-\
            CYCLE_DATA["start_date"][-1][0])*12 + (CYCLE_DATA["end_date"][-1][1]-\
            CYCLE_DATA["start_date"][-1][1]))

            CYCLE_DATA["start_date"].append([curr_min[0], curr_min[1]])
            curr_min[2] = 500
        if year > curr_max[0] + 5:
            CYCLE_DATA["max_date"].append([curr_max[0], curr_max[1]])
            CYCLE_DATA["solar_max"].append(curr_max[2])
            curr_max[2] = 0

        if ssn <= curr_min[2]:
            curr_min = [year, month, ssn]

        if ssn >= curr_max[2]:
            curr_max = [year, month, ssn]

    CYCLE_DATA["end_date"].append([year + (month-1)//12, (month-1)%12])
    CYCLE_DATA["length"].append((CYCLE_DATA["end_date"][-1][0]-\
    CYCLE_DATA["start_date"][-1][0])*12 + (CYCLE_DATA["end_date"][-1][1]-\
    CYCLE_DATA["start_date"][-1][1]))

    cycle_file = open("cycle_data.pickle", "wb")
    pickle.dump(CYCLE_DATA, cycle_file)
    cycle_file.close()

    return CYCLE_DATA


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
