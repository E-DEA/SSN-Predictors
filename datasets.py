#!/usr/bin/env python3

import os
import math

import numpy as np

from torch.utils.data import Dataset
from sympy import pi as PI

#Earliest cycle data that is used for prediction.
START_CYCLE = 12

#Latest cycle data used for predictions.
END_CYCLE = 24

class Features(Dataset):
    def __init__(self, SSN_data, AA_data, cycle_data, start_cycle=START_CYCLE, end_cycle=END_CYCLE):
        super(Features, self).__init__()

        self.features = []
        self.targets = []
        self.ssn = SSN_data
        self.aa = AA_data

        self.__gen_samples(cycle_data, start_cycle, end_cycle)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (self.features[index], self.targets[index])

    def __gen_samples(self, CYCLE_DATA, start_cycle, end_cycle):
        start_date = CYCLE_DATA["start_date"][start_cycle]
        end_date = CYCLE_DATA["end_date"][end_cycle]

        self.targets += self.ssn.data[start_date[0]][start_date[1]-1:]

        for year in range(start_date[0] + 1, end_date[0]):
            self.targets += self.ssn.data[year]

        self.targets += self.ssn.data[end_date[0]][:end_date[1]]

        for cycle in range(start_cycle, end_cycle + 1):
            start_date = CYCLE_DATA["start_date"][cycle]
            end_date = CYCLE_DATA["end_date"][cycle]

            tf = (end_date[0] - start_date[0])*12 + (end_date[1] - start_date[1] + 1)

            start_date = CYCLE_DATA["start_date"][cycle - 1]
            end_date = CYCLE_DATA["end_date"][cycle - 1]

            for step in range(tf):
                month_num = (step % 12) + 1
                year_index  = (step // 12) + 1

                month = (start_date[1] + month_num - 1) % 12
                year = (start_date[0] + year_index - 1) + (start_date[1] + month_num - 1)//12

                delayed_aa = self.aa.data[year][month]
                delayed_ssn = self.ssn.data[year][month]

                ms = math.sin((2*PI*month_num)/12)
                mc = math.cos((2*PI*month_num)/12)
                ys = math.sin((2*PI*year_index)/11)
                yc = math.cos((2*PI*year_index)/11)

                self.features.append(np.array([ys, yc, ms, mc, delayed_aa, delayed_ssn]))

class AA(Dataset):
    def __init__(self, file):
        super(AA, self).__init__()

        self.file = file
        self.data = {}

        self._get_data()

        self.yeardata = list(self.data.keys())
        self.valdata = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.yeardata[index], self.valdata[index])

    def __extract_data(self):
        read = False
        dates = []
        curr_aa = []
        aa_val = 0.0
        num_days = 0
        with open(self.file, "r") as fp:
            for line in fp:
                terms = line.split()

                if not terms:
                    continue

                date = terms[0]

                if date == "DATE":
                    read = True
                    continue

                if read:
                    datetuple = date.split("-")
                    year = int(datetuple[0])
                    month = int(datetuple[1])

                    if year not in self.data.keys():
                        self.data[year] = []

                    if datetuple in dates:
                        continue
                    else:
                        if dates and int(dates[-1][1]) != month:
                            aa_val /= num_days
                            curr_aa.append(aa_val)
                            aa_val = 0.0
                            num_days = 0

                            if int(dates[-1][0]) != year:
                                self.data[int(dates[-1][0])] = curr_aa
                                curr_aa = []
                                dates = []

                        dates.append(datetuple)
                        aa_val += float(terms[5])
                        num_days += 1

            aa_val /= num_days
            curr_aa.append(aa_val)
            self.data[year] = curr_aa

    def _get_data(self):
        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()

class SSN(Dataset):
    def __init__(self, file):
        super(SSN, self).__init__()

        self.file = file

        self.data = {}

        self._get_data()

        self.yeardata = list(self.data.keys())
        self.valdata = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.yeardata[index], self.valdata[index])

    def __extract_data(self):
        read = True
        curr_MSN = []
        with open(self.file, "r") as fp:
            for line in fp:
                terms = line.split()
                year = int(terms[0])

                if (float(terms[3]) < 0):
                    continue

                if year not in self.data.keys():
                    self.data[year] = []

                self.data[year].append(float(terms[3]))

    def _get_data(self):
        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()
