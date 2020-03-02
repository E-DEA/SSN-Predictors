#!/usr/bin/env python3

import os
import math

import numpy as np
import utility as ut

from torch.utils.data import Dataset

PI = math.pi

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",\
"Nov", "Dec"]

CYCLE_DATA = {"start_date":[], "end_date":[], "max_date":[], "solar_max":[], "length":[]}

seed = 1

start_cycle = 12
end_cycle = 23

np.random.seed(seed)

class Features(Dataset):
    def __init__(self, SSN_data, AA_data):
        super(Features, self).__init__()

        self.features = []
        self.targets = []
        self.ssn = SSN_data
        self.aa = AA_data

        self.__gen_samples()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (self.features[index], self.targets[index])

    def __gen_samples(self):
        start_date = CYCLE_DATA[start_cycle + 1]["start_date"]
        end_date = CYCLE_DATA[end_cycle]["end_date"]

        self.targets.append(self.ssn.data[start_date[0]][start_date[1]-1:])

        for year in range(start_date[0] + 1, end_date[0]):
            self.targets.append(self.ssn.data[year])

        self.targets.append(self.ssn.data[end_date[0]][:end_date[1]])

        temp_feats = []

        for cycle in range(start_cycle + 1, end_cycle + 1):
            start_date = CYCLE_DATA[cycle]["start_date"]
            end_date = CYCLE_DATA[cycle]["end_date"]

            for month in range(start_date[1], 13):
                ms = math.sin((2*PI*month)/12)
                mc = math.cos((2*PI*month)/12)

                temp_feats.append([0.0, 1.0, ms, mc])

            for year in range(start_date[0] + 1, end_date[0]):
                for month in range(1, 13):
                    year_index = ((month - start_date[1]) + (year - start_date[0])*12)//12

                    ms = math.sin((2*PI*month)/12)
                    mc = math.cos((2*PI*month)/12)
                    ys = math.sin((2*PI*year_index)/11)
                    yc = math.cos((2*PI*year_index)/11)

                    temp_feats.append([ys, yc, ms, mc])

            for month in range(1, end_date[1] + 1):
                year_index = ((month - end_date[1]) + (end_date[0] - start_date[0])*12)//12


class AA(Dataset):
    def __init__(self, file):
        super(AA, self).__init__()

        self.file = file
        self.data = {}

        self._get_data()

        self.__yeardata = list(self.data.keys())
        self.__valdata = list(self.data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.__yeardata[index], self.__valdata[index])

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
                                self.data[int(dates[-1][0])].append(curr_aa)
                                curr_aa = []
                                dates = []

                        dates.append(datetuple)
                        aa_val += float(terms[5])
                        num_days += 1

            aa_val /= num_days
            curr_aa.append(aa_val)
            self.data[year].append(curr_aa)

    def _get_data(self):
        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()

class SSN(Dataset):
    global CYCLE_DATA
    def __init__(self, file):
        super(SSN, self).__init__()

        self.file = file

        self.data = {}

        self._get_data()

        self.__yeardata = list(self.data.keys())
        self.__valdata = list(self.data.values())

        self._get_cycles()

        #print(CYCLE_DATA)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.__yeardata[index], self.__valdata[index])

    def __extract_data(self):
        read = True
        curr_MSN = []
        with open(self.file, "r") as fp:
            for line in fp:
                terms = line.split()
                year = int(terms[0])

                if year not in self.data.keys():
                    self.data[year] = []

                self.data[year].append(float(terms[3]))

    def _get_data(self):
        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()

    def _get_cycles(self):
        data = []
        for idx, year in enumerate(self.__yeardata):
            for month, ssn in enumerate(self.__valdata[idx]):
                data.append(float(ssn))

        data = ut.sidc_filter(data)

        curr_min = [self.__yeardata[0], 1, 500]
        curr_max = [self.__yeardata[0], 1, 0]
        CYCLE_DATA["start_date"].append([self.__yeardata[0], 1])

        for idx, ssn in enumerate(data):
            year = self.__yeardata[0] + idx//12
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
