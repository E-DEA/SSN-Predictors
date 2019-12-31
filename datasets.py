#!/usr/bin/env python3

import os
import math

import numpy as np
import utility as ut

from torch.utils.data import Dataset

PI = math.pi

VALID_SOURCES = ["NOAA", "SILSO"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",\
"Nov", "Dec"]

CYCLE_DATA = {"start_date":[], "end_date":[], "max_date":[], "solar_max":[], "length":[]}

seed = 1

start_cycle = 13
end_cycle = 23

np.random.seed(seed)

class Features(Dataset):
    def __init__(self, *datasets):
        super(Features, self).__init__()

        self.samples = []
        self.datasets = datasets

        self.__gen_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (self.samples[index])

    def __gen_samples(self):
        for curr_cycle in range(start_cycle, end_cycle + 1):
            start_date = CYCLE_DATA["start_date"][curr_cycle]
            end_date = CYCLE_DATA["end_date"][curr_cycle]

            year_index = []
            for dataset in self.datasets:
                idx = dataset.data["years"].index(CYCLE_DATA["start_date"][curr_cycle-1][0])
                year_index.append(idx)

            for curr_year in range(start_date[0], end_date[0] + 1):
                yi = curr_year - start_date[0]

                ys = math.sin((2*PI*yi)/11)
                yc = math.cos((2*PI*yi)/11)

                curr_idx = dataset.data["years"].index(curr_year)

                for curr_month in range(len(MONTHS)):
                    if (curr_month + 1 < start_date[1] and curr_year == start_date[0])\
                    or (curr_month + 1 > end_date[1] and curr_year == end_date[0]):
                        continue

                    ms = math.sin((2*PI*curr_month)/12)
                    mc = math.cos((2*PI*curr_month)/12)

                    feat = [ys, yc, ms, mc]

                    for dnum, dataset in enumerate(self.datasets):
                        idx = year_index[dnum]
                        val = dataset.data["vals"][idx][curr_month]
                        feat.append(val)

                        if dataset.__class__.__name__ == "SSN":
                            target = dataset.data["vals"][curr_idx][curr_month]

                    self.samples.append((np.array(feat), target))

                for idx in year_index:
                    idx += 1


class AA(Dataset):
    def __init__(self, file):
        super(AA, self).__init__()

        self.file = file
        self.data = {"years":[], "vals":[]}
        self.cycle_data = []

        self._get_data()

    def __len__(self):
        return len(self.data["years"])

    def __getitem__(self, index):
        return (self.data["years"][index], self.data["vals"][index])

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

                    if datetuple in dates:
                        continue
                    else:
                        if dates and dates[-1][1] != datetuple[1]:
                            aa_val /= num_days
                            curr_aa.append(aa_val)
                            aa_val = 0.0
                            num_days = 0

                            if dates[-1][0] != datetuple[0]:
                                if len(curr_aa)==12:
                                    self.data["vals"].append(curr_aa)
                                    self.data["years"].append(int(dates[-1][0]))
                                curr_aa = []
                                dates = []

                        dates.append(datetuple)
                        aa_val += float(terms[5])
                        num_days += 1

            aa_val /= num_days
            curr_aa.append(aa_val)
            if len(curr_aa)==12:
                self.data["vals"].append(curr_aa)
                self.data["years"].append(int(dates[-1][0]))

    def _get_data(self):
        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()

class SSN(Dataset):
    global CYCLE_DATA
    def __init__(self, source, file):
        super(SSN, self).__init__()

        self.source = source
        self.file = file

        self.data = {"years":[], "vals":[]}

        self._get_data()
        self._get_cycles()

        print(CYCLE_DATA)

    def __len__(self):
        return len(self.data["years"])

    def __getitem__(self, index):
        return (self.data["years"][index], self.data["vals"][index])

    def __extract_data(self):
        if self.source == "NOAA":
            read = False
            with open(self.file, "r") as fp:
                for line in fp:
                    terms = line.split()

                    if not terms:
                        continue

                    year = terms[0]

                    if year == "Year":
                        read = True
                        skip = True
                    else:
                        if not read or skip:
                            skip = False
                            continue
                        else:
                            if not skip and len(year)!=4:
                                read = False
                            else:
                                self.data["years"].append(int(year))
                                self.data["vals"].append([float(i) for i in terms[1:]])
        elif self.source == "SILSO":
            read = True
            curr_MSN = []
            with open(self.file, "r") as fp:
                for line in fp:
                    terms = line.split()
                    year = int(terms[0])

                    if year not in self.data["years"]:
                        if not curr_MSN:
                            pass
                        else:
                            self.data["vals"].append(curr_MSN)
                            curr_MSN = []
                        self.data["years"].append(year)
                        curr_MSN.append(float(terms[3]))
                    else:
                        curr_MSN.append(float(terms[3]))

                self.data["vals"].append(curr_MSN)

    def _get_data(self):
        if self.source not in VALID_SOURCES:
            print("Given source: {} is not a valid source".format(self.source))
            return -1

        if not os.path.isfile(self.file):
            print("File Error: No such file {}".format(self.file))
            return -1

        self.__extract_data()

    def _get_cycles(self):
        data = []
        for idx, year in enumerate(self.data["years"]):
            for month, ssn in enumerate(self.data["vals"][idx]):
                data.append(float(ssn))

        data = ut.sidc_filter(data)

        curr_min = [self.data["years"][0], 1, 500]
        curr_max = [self.data["years"][0], 1, 0]
        CYCLE_DATA["start_date"].append([self.data["years"][0], 1])

        for idx, ssn in enumerate(data):
            year = self.data["years"][0] + idx//12
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
