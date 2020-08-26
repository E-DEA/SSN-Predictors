#!/usr/bin/env python3

import os
import datasets

import datetime as dt

from pandas import plotting as pltng
from matplotlib import pyplot as plt
from matplotlib import dates as dts

pwd = os.getcwd()

lossfolder = pwd + "/graphs/loss/"
predfolder = pwd + "/graphs/ssn/"

LINESPLIT = "-" * 64

pltng.register_matplotlib_converters()

def plot_loss(label, steps, loss, filename):
    print(LINESPLIT)
    print("Plotting loss data...")

    plt.plot(steps, loss, label=label, aa=True)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(lossfolder+filename, dpi=600)


def plot_predictions(label, xdata, ydata, filename, compare=False):
    print(LINESPLIT)
    print("Plotting data...")

    fig, ax = plt.subplots()

    if compare:
        source = "data/SILSO/TSN/SN_m_tot_V2.0.txt"
        source_data = datasets.SSN(source)

        source_xdata = []
        source_ydata = []

        for idx, year in enumerate(source_data.yeardata):
            if year >= xdata[0].year[0] and year <= xdata[-1].year[0]:
                for month, val in enumerate(source_data.valdata[idx]):
                    source_xdata.append(dt.datetime(year=year, month=month+1, day=15))
                    source_ydata.append(float(val))

        ax.plot_date(source_xdata, source_ydata, "-m", xdate=True, label="SILSO", lw=0.5, aa=True)
        ax.set_ylabel("Monthly SSN")

    ax.plot_date(xdata, ydata, "--b", xdate=True, label=label, lw=0.75, aa=True)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    plt.legend()

    majortick = dts.YearLocator(5)
    minortick = dts.YearLocator(1)
    ticker_fmt = dts.DateFormatter("%Y")

    ax.xaxis.set_major_locator(majortick)
    ax.xaxis.set_major_formatter(ticker_fmt)
    ax.xaxis.set_minor_locator(minortick)

    plt.savefig(predfolder+filename, dpi=240)
    plt.close("all")

def plot_all(savefile):

    print(LINESPLIT)
    print("Plotting data...")

    data_source = "SILSO"
    aa_source = "ISGI"

    data_file = "data/SILSO/TSN/SN_m_tot_V2.0.txt"
    aa_file = "data/ISGI/aa_1869-01-01_2018-12-31_D.dat"

    data1 = datasets.SSN(data_file)
    data2 = datasets.AA(aa_file)

    xdata1 = []
    ydata1 = []
    xdata2 = []
    ydata2 = []

    for idx, year in enumerate(data1.yeardata):
        for month, ssn in enumerate(data1.valdata[idx]):
            xdata1.append(dt.datetime(year=year, month=month+1, day=15))
            ydata1.append(float(ssn))

    for idx, year in enumerate(data2.yeardata):
        for month, aa in enumerate(data2.valdata[idx]):
            xdata2.append(dt.datetime(year=year, month=month+1, day=15))
            ydata2.append(float(aa))

    ysmoothed1 = ut.sidc_filter(ydata1)
    ysmoothed2 = ut.sidc_filter(ydata2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(36, 9), sharex="col")

    ax1.plot_date(xdata1, ydata1, "-b", xdate=True, label=data_source, lw=0.5, aa=True)
    ax1.plot_date(xdata1, ysmoothed1, "--b", xdate=True, alpha=0.75, lw=1, aa=True)

    ax1.set_ylabel("Monthly Sunspot Number")

    ax2.plot_date(xdata2, ydata2, "-m", xdate=True, label=aa_source, lw=0.5, aa=True)
    ax2.plot_date(xdata2, ysmoothed2, "--m", xdate=True, alpha=0.75, lw=1, aa=True)

    ax2.set_xlabel("Year")
    ax2.set_ylabel("AA Index")

    ax1.legend()
    ax2.legend()

    majortick = dts.YearLocator(10)
    minortick = dts.YearLocator(2)
    ticker_fmt = dts.DateFormatter("%Y")

    ax2.xaxis.set_major_locator(majortick)
    ax2.xaxis.set_major_formatter(ticker_fmt)
    ax2.xaxis.set_minor_locator(minortick)

    plt.savefig(graphfolder+savefile, dpi=240)
    plt.close("all")

    print(LINESPLIT)
    print("Data plots saved in {} as '{}'".format(graphfolder, savefile))

def main():
    savefile = input("Enter the savefile name for the plot: ")
    plot_all(savefile)

if __name__ == '__main__':
    main()
