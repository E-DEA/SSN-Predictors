#!/usr/bin/env python3

import os
import sys
import math
import random
import torch
import models
import datasets

import utility as ut
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import pyplot as plt
from matplotlib import dates as dts

pwd = os.getcwd()

graphfolder = pwd + "/graphs/"
modelfolder = pwd + "/models/"
logfolder = pwd + "/logs/"

LINESPLIT = "-" * 50

sys.stdout = ut.Logger("{}{}.log".format(logfolder, dt.datetime.now()))

seed = 1

random.seed(seed)
torch.manual_seed(seed)

MAX_EPOCHS = 10000
BATCH_SIZE = 3

epochs = 150 * BATCH_SIZE
learning_rate = 0.0005

PRINT_FREQ = 1
SAVE_FREQ = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pd.plotting.register_matplotlib_converters()

def print_loss(xdata, ydata, xlabel, ylabel, filename):
    plt.plot(xdata, ydata, lw=0.2, aa=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(graphfolder+filename, dpi=900)
    plt.close("all")

def print_data(pred_msn, msn, filename):
    xdata = []
    ydata1 = []
    ydata2 = []
    for sample_num, (pred_ssn, ssn) in enumerate(zip(pred_msn, msn)):
        xdata.append(dt.datetime(year=datasets.start_year + sample_num//12, month=(sample_num % 12) + 1, day=1))
        ydata1.append(float(ssn))
        ydata2.append(float(pred_ssn))

    plt.figure(figsize=(19.2, 4.8))
    plt.plot(xdata, ydata1, "-",label="Observed", lw=0.2, aa=True)
    plt.plot(xdata, ydata2, "--",label="Predicted", lw=0.4, aa=True)
    plt.xlabel("Year")
    plt.ylabel("Monthly Sunspot Number")
    plt.legend()
    plt.savefig(graphfolder+filename, dpi=900)
    plt.close("all")


def load_model(model, folder=modelfolder):
    for i in range(MAX_EPOCHS, SAVE_FREQ, -SAVE_FREQ):
        try:
            model.load_state_dict(torch.load("{}_{}_{}.pth".format(modelfolder, model.__class__.__name__, i)))
            print(LINESPLIT)
            print("Pre-trained model available, loading model weights")
            return True
        except:
            pass

    print(LINESPLIT)
    print("No pre-trained models available, initializing model weights")
    model.apply(ut.weight_init)

    return False

def train(model, train_loader, optim, sch, num_epochs):
    avg_loss = []
    for epoch in range(num_epochs):
        if epoch == MAX_EPOCHS:
            print(LINESPLIT)
            print("Maximum allowed training epochs({}) reached".format(epoch))

        total_steps = len(train_loader)
        running_loss = 0.0

        for step, (feats, sns) in enumerate(train_loader):
            feats = feats.to(device)
            sns = sns.to(device)

            optim.zero_grad()

            outputs = model(feats.float())
            loss = models.criterion(outputs.squeeze(1), sns.float())

            loss.backward()
            optim.step()

            running_loss += loss.item()

        avg_loss.append(running_loss/total_steps)

        sch.step(running_loss/total_steps)

        if (epoch+1) % SAVE_FREQ == 0:
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(modelfolder, model.__class__.__name__, epoch+1))
            print(LINESPLIT)
            print("Model checkpoint saved as {}_{}.pth".format(model.__class__.__name__, epoch+1))
            print(LINESPLIT)

        if (epoch+1) % PRINT_FREQ == 0:
            print("Epoch [{:4d}/{}] -> Loss: {:.4f}".format(epoch+1, num_epochs, running_loss/total_steps))
            running_loss = 0.0

    return avg_loss

def test(model, test_loader):
    with torch.no_grad():
        predictions = []
        for (feats, sns) in test_loader:
            feats = feats.to(device)
            sns = sns.to(device)

            predictions.append(model(feats.float()))

    return predictions

def predict(model, obs_data, pred_window):
    with torch.no_grad():
        for idx in range(pred_window):
            yi = idx//12 + (datasets.end_year - datasets.start_year + 1)
            ys = math.sin(((2*PI)*yi)/11)
            yc = math.sin(((2*PI)*yi)/11)

            month = (idx % 12) + 1
            ms = math.sin(((2*PI)*month)/12)
            mc = math.cos(((2*PI)*month)/12)


"""
Driver code to run the predictor.
"""
def main(is_train, is_test, predict):
    if len(sys.argv) < 4:
        print("Usage: python3 {} <source_name> <path_to_ssn_datafile> <path_to_aa_datafile>".format(os.path.basename(__file__)))
        return

    print(LINESPLIT)
    print("Code running on device: {}".format(device))

    data_source = sys.argv[1]
    aa_source = "ISGI"
    data_file = sys.argv[2]
    aa_file = sys.argv[3]

    ssn_data = datasets.SSN(data_source, data_file)
    aa_data = datasets.AA(aa_file)

    train_samples = datasets.Features(ssn_data, aa_data)

    print(train_samples.samples[:50])

    return

    print(LINESPLIT)
    print("Dataset source : {}, {}".format(data_source, aa_source))
    print('''File location :
    SSN - {}
    AA - {}'''.format(os.path.abspath(data_file), os.path.abspath(aa_file)))

    ######## FFNN ########

    model = models.FFNN(inp_dim=6).to(device)

    print(LINESPLIT)
    print('''Selected model: {}\
    Training mode: {}\
    Testing mode: {}'''.format(model, is_train, is_test))

    pre_trained = load_model(model)

    if not pre_trained:
        if not is_train and (is_test or predict):
            print(LINESPLIT)
            print("Warning: Testing/Prediction is ON with training OFF and no pretrained model available")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, verbose=True)

    train_loader = DataLoader(dataset=train_samples, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=train_samples, batch_size=1, shuffle=False)

    if is_train:
        model.train()
        print(LINESPLIT)
        print("Training model with: num_epochs={}, start_lr={}".format(epochs, learning_rate))

        loss = train(model, train_loader, optimizer, scheduler, epochs)
        torch.save(model.state_dict(), "{}_{}_{}.pth".format(modelfolder, model.__class__.__name__, MAX_EPOCHS))

        print(LINESPLIT)
        print('''Training finished successfully.
        Saved model checkpoints can be found in: {}
        Saved data/loss graphs can be found in: {}'''.format(modelfolder, graphfolder))

        print_loss(range(len(loss)),loss, "Epochs", "avg. loss", "loss/tr_{}_{}.png".format(model.__class__.__name__, data_source))

    if is_test:
        model.eval()
        print(LINESPLIT)
        print("Testing model")
        predictions = test(model, test_loader)

        print_data(predictions, train_samples.targets, "ssn/{}_{}.png".format(model.__class__.__name__, data_source))

    ######## RNN ########

def plot_data():

    print(LINESPLIT)
    print("Plotting data...")

    data_source1 = "NOAA"
    data_source2 = "SILSO"

    data_file1 = "/home/extern/Documents/Research/data/NOAA/table_international-sunspot-numbers_monthly.txt"
    data_file2 = "/home/extern/Documents/Research/data/SILSO/TSN/SN_m_tot_V2.0.txt"
    aa_file = "/home/extern/Documents/Research/data/ISGI/aa_1869-08-01_2017-12-31_D.dat"

    data1 = datasets.SSN(data_source1, data_file1)
    data2 = datasets.SSN(data_source2, data_file2)
    data3 = datasets.AA(aa_file)

    xdata1 = []
    ydata1 = []
    xdata2 = []
    ydata2 = []
    xdata3 = []
    ydata3 = []

    for idx, year in enumerate(data1.data["years"]):
        for month, ssn in enumerate(data1.data["vals"][idx]):
            xdata1.append(dt.datetime(year=year, month=month+1, day=1))
            ydata1.append(float(ssn))

    for idx, year in enumerate(data2.data["years"]):
        for month, ssn in enumerate(data2.data["vals"][idx]):
            xdata2.append(dt.datetime(year=year, month=month+1, day=1))
            ydata2.append(float(ssn))

    for idx, year in enumerate(data3.data["years"]):
        for month, aa in enumerate(data3.data["vals"][idx]):
            xdata3.append(dt.datetime(year=year, month=month+1, day=1))
            ydata3.append(float(aa))

    ysmoothed1 = ut.sidc_filter(ydata1)
    ysmoothed2 = ut.sidc_filter(ydata2)
    ysmoothed3 = ut.sidc_filter(ydata3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 9.6), sharex="col")

    ax1.plot(xdata1, ydata1, "b", label=data_source1, lw=0.2, aa=True)
    ax1.plot(xdata1, ysmoothed1, "--b", lw=1, aa=True)

    ax1.plot(xdata2, ydata2, "c", label=data_source2, lw=0.2, aa=True)
    ax1.plot(xdata2, ysmoothed2, "--c", lw=1, aa=True)

    ax1.set_ylabel("Monthly Sunspot Number")

    ax2.bar(xdata3, ydata3, color="m", width=4)
    ax2.plot(xdata3, ysmoothed3, "--m", label="ISGI", lw=1, aa=True)

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

    plt.savefig(graphfolder+"combined_data.png", dpi=900)
    plt.close("all")

    print(LINESPLIT)
    print("Data plots saved in {} as '{}'".format(graphfolder, "combined_data.png"))

if __name__=="__main__":
    is_train = True
    is_test = False
    predict = False
    #plot_data()
    main(is_train, is_test, predict)
