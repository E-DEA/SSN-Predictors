#!/usr/bin/env python3

import sys
import os

import utility as ut

from datasets import START_CYCLE, END_CYCLE, SSN, AA
from numpy import mean, std, cov
from scipy.stats import spearmanr, pearsonr

# IMPORTANT:
# xdata always remains the same, i.e. the target SSN values.
# ydata changes to whatever data we are analyzing against it
# ONLY ONE of either time delay and month delay should be active at a time.


# time delay if checking with an integral delay of solar cycle.
time_delay = 0

# month delay if checking with an integral delay of months(steps).
month_delay = 12

LINESPLIT = "-" * 100

def main():
    if len(sys.argv) < 2:
        print(LINESPLIT)
        print("Usage: python data_analysis.py <dataset to be compared to SSN for analysis>")
        return

    xfile = "data/SILSO/TSN/SN_m_tot_V2.0.txt"
    yfile = sys.argv[1]
    xdataset = SSN(xfile)

    if "SILSO" in yfile:
        ydataset = SSN(yfile)
    elif "ISGI" in yfile:
        ydataset = AA(yfile)

    print(LINESPLIT)
    print('''Data loaded from file locations :
    SSN Target - {}
    Precursor - {}'''.format(os.path.abspath(xfile), os.path.abspath(yfile)))

    CYCLE_DATA = ut.get_cycles(xdataset)

    print(LINESPLIT)
    print("Solar cycle data loaded/saved as: cycle_data.pickle")
    print(LINESPLIT)
    ut.print_cycles(CYCLE_DATA)

    if time_delay > 0:
        xdata = []
        ydata = []
        for cycle in range(START_CYCLE + time_delay, END_CYCLE + 1):
            xstart_date = CYCLE_DATA["start_date"][cycle]
            xend_date = CYCLE_DATA["end_date"][cycle]

            ystart_date = CYCLE_DATA["start_date"][cycle - time_delay]
            yend_date = CYCLE_DATA["end_date"][cycle - time_delay]

            xtemp = xdataset.data[xstart_date[0]][xstart_date[1]-1:]
            ytemp = ydataset.data[ystart_date[0]][ystart_date[1]-1:]

            for year in range(xstart_date[0] + 1, xend_date[0]):
                xtemp += xdataset.data[year]

            for year in range(ystart_date[0] + 1, yend_date[0]):
                ytemp += ydataset.data[year]

            xtemp += xdataset.data[xend_date[0]][:xend_date[1]]
            ytemp += ydataset.data[yend_date[0]][:yend_date[1]]

            datadiff = len(xtemp) - len(ytemp)

            if datadiff > 0:
                ytemp += xtemp[-datadiff:]
            elif datadiff < 0:
                xtemp += ytemp[datadiff:]

            xdata += xtemp
            ydata += ytemp
    else:
        start_date = CYCLE_DATA["start_date"][START_CYCLE]
        end_date = CYCLE_DATA["end_date"][END_CYCLE]

        xdata = xdataset.data[start_date[0]][start_date[1]-1:]
        ydata = ydataset.data[start_date[0]][start_date[1]-1:]

        for year in range(start_date[0] + 1, end_date[0]):
            xdata += xdataset.data[year]
            ydata += ydataset.data[year]

        xdata += xdataset.data[end_date[0]][:end_date[1]]
        ydata += ydataset.data[end_date[0]][:end_date[1]]

    if month_delay > 0:
        xdata = xdata[month_delay:]
        ydata = ydata[:-month_delay]

    print(LINESPLIT)
    print('''Data ready for data analysis.
    Start Cycle: {}
    End Cycle: {}
    Time delay: {}
    Month/Step delay: {}
    X-axis data: SSN, {} datapoints
    Y-axis data: {}, {} datapoints'''.format(START_CYCLE, END_CYCLE,\
    time_delay, month_delay, len(xdata), ydataset.__class__.__name__,len(ydata)))

    covariance = cov(xdata, ydata)
    PearsonsCorr, _ = pearsonr(xdata, ydata)
    SpearmansCorr, _ = spearmanr(xdata, ydata)

    print(LINESPLIT)
    print('''Calculated correlation coefficients are as follows:
    Covariance Matrix: {}
    Pearsons Correlation: {}
    Spearmans Correlation: {}'''.format(covariance, PearsonsCorr, SpearmansCorr))

if __name__ == '__main__':
    print(LINESPLIT)
    print('''\
    Running data analysis and finding covariance, Pearson coefficient
    and Spearman coefficient between the SSN SILSO dataset and the dataset provided.''')

    main()
