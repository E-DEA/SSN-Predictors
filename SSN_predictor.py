#!/usr/bin/env python3

import os
import sys
import torch
import models
import datasets
import plotter

import utility as ut
import datetime as dt

from pandas import plotting as pltng
from torch.utils.data import DataLoader

pwd = os.getcwd()

graphfolder = pwd + "/graphs/"
modelfolder = pwd + "/models/"
logfolder = pwd + "/logs/"

LINESPLIT = "-" * 100

sys.stdout = ut.Logger("{}{}.log".format(logfolder, dt.datetime.ctime(dt.datetime.now())))
sys.stderr = sys.stdout

seed = 1

torch.manual_seed(seed)

MAX_EPOCHS = 10000
BATCH_SIZE = 4

epochs = 800
learning_rate = 0.001
eps = 1e-6

PRINT_FREQ = 5
SAVE_FREQ = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pltng.register_matplotlib_converters()

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
            loss = torch.sqrt(models.criterion(outputs.squeeze(1), sns.float()) + eps)

            loss.backward()
            optim.step()

            running_loss += loss.item()

        avg_loss.append(running_loss/total_steps)

        sch.step(running_loss/total_steps)

        if (epoch+1) % SAVE_FREQ == 0:
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(modelfolder, model.__class__.__name__, epoch+1))
            print(LINESPLIT)
            print("Model checkpoint saved as _{}_{}.pth".format(model.__class__.__name__, epoch+1))
            print(LINESPLIT)

        if (epoch+1) % PRINT_FREQ == 0:
            print("Epoch [{:4d}/{}] -> Loss: {:.4f}".format(epoch+1, num_epochs, running_loss/total_steps))
            running_loss = 0.0

    return avg_loss

def validate(model, val_loader, timestamps):
    predictions = []
    total_loss = []
    running_loss = 0.0

    with torch.no_grad():
        for step, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)

            prediction = model(data.float())
            loss = torch.sqrt(models.criterion(prediction.squeeze(1), target.float()))

            print("Step [{:4d}/{}] -> Date: {}/{}, Target: {}, Prediction: {}".\
            format(step+1, len(val_loader), timestamps[step].month, timestamps[step].year,\
            target.item(), prediction.item()))

            predictions.append(float(prediction.item()))
            total_loss.append(loss.item())

            running_loss += loss.item()

    print(LINESPLIT)
    print("Average Validation Loss: {}".format(running_loss/len(total_loss)))

    return (predictions, total_loss)

def predict(model, pred_feats, timestamps):
    predictions = []

    with torch.no_grad():
        for step, data in enumerate(pred_feats):
            data = torch.tensor(data).to(device)

            prediction = model(data.float())

            print("Step [{:4d}/{}] -> Date: {}/{}, Prediction: {}".\
            format(step+1, len(pred_feats), timestamps[step].month, timestamps[step].year,\
            prediction.item()))

            predictions.append(float(prediction.item()))

    return predictions

"""
Driver code to run the predictor.
"""
def main(is_train, prediction, plotting, scaling, selected_model):
    if len(sys.argv) < 3:
        print(LINESPLIT)
        print("Usage: python3 {} <path_to_ssn_datafile> <path_to_aa_datafile>".format(os.path.basename(__file__)))
        data_file = "data/SILSO/TSN/SN_m_tot_V2.0.txt"
        aa_file = "data/ISGI/aa_1869-01-01_2020-12-19_D.dat"
    else:
        data_file = sys.argv[1]
        aa_file = sys.argv[2]

    print(LINESPLIT)
    print("Code running on device: {}".format(device))

    ssn_data = datasets.SSN(data_file)
    aa_data = datasets.AA(aa_file)

    print(LINESPLIT)
    print('''Data loaded from file locations :
    SSN - {}
    AA - {}'''.format(os.path.abspath(data_file), os.path.abspath(aa_file)))

    if plotting:
        plotter.plot_all("combined_data1.jpg")

    cycle_data = ut.get_cycles(ssn_data)

    print(LINESPLIT)
    print("Solar cycle data loaded/saved as: cycle_data.pickle")
    print(LINESPLIT)
    ut.print_cycles(cycle_data)

    train_samples = datasets.Features(ssn_data, aa_data, cycle_data, normalize=scaling,\
    start_cycle=13, end_cycle=22)
    valid_samples = datasets.Features(ssn_data, aa_data, cycle_data, normalize=scaling,\
    start_cycle=23, end_cycle=23)
    valid_timestamps, _ = ut.gen_samples(ssn_data, aa_data, cycle_data,\
    cycle=23, normalize=scaling, tf=cycle_data["length"][23])
    predn_timestamps, predn_samples = ut.gen_samples(ssn_data, aa_data, cycle_data,\
    cycle=24, normalize=scaling)

    print(LINESPLIT)
    print('''Selected data:
    Training: SC 13 to 22
    Validation: SC 23
    Prediction: SC 24''')

    ############ FFNN/RNN/LSTM (model chosen by user) ############

    model = getattr(models, selected_model)(inp_dim=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, verbose=True)

    print(LINESPLIT)
    print('''Selected model: {}\
    Training mode: {}\
    Prediction mode: {}'''.format(model, is_train, prediction))

    print(LINESPLIT)
    print("Selected optimizer: {}".format(optimizer))

    print(LINESPLIT)
    print('''Selected scheduler: {}(
    {})'''.format(scheduler.__class__.__name__, scheduler.state_dict()))

    pre_trained = load_model(model)

    if not pre_trained:
        if not is_train and prediction:
            print(LINESPLIT)
            print("Warning: Prediction is ON with training OFF and no pretrained models available")

    train_loader = DataLoader(dataset=train_samples, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_samples, batch_size=1, shuffle=False)

    ### Training ###

    if is_train:
        if not pre_trained:
            model.train()
            print(LINESPLIT)
            print("Training model with solar cycle {} to {} data with: num_epochs={}".\
            format(datasets.START_CYCLE, datasets.END_CYCLE - 2, epochs))

            loss = train(model, train_loader, optimizer, scheduler, epochs)
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(modelfolder, model.__class__.__name__, MAX_EPOCHS))

            plotter.plot_loss("Average Training Loss", range(len(loss)), loss, "tr_{}.png".\
            format(model.__class__.__name__))

            print(LINESPLIT)
            print('''Training finished successfully.
            Saved model checkpoints can be found in: {}
            Saved data/loss graphs can be found in: {}'''.format(modelfolder, graphfolder))

        else:
            print(LINESPLIT)
            print("Skipping training, using pre-trained model for validation and prediction")

    ### Validating ###

        model.eval()
        print(LINESPLIT)
        print("Validating model for solar cycle {} data".format(datasets.END_CYCLE - 1))

        valid_predictions, valid_loss = validate(model, valid_loader, valid_timestamps)

        plotter.plot_predictions("SC{} Prediction".format(datasets.END_CYCLE - 1),\
        valid_timestamps, valid_predictions, "SC 23 Validation.png", compare=True)
        plotter.plot_loss("Validation Loss", range(len(valid_loss)), valid_loss, "val_{}.png".\
        format(model.__class__.__name__))

        print(LINESPLIT)
        print('''Validation finished successfully.\n
        Saved prediction/loss graphs can be found in: {}'''.format(graphfolder))

    ### Predicting ###

    if prediction:
        model.eval()
        print(LINESPLIT)
        print("Predicting SC {} using the above trained model".format(datasets.END_CYCLE))

        predn_predictions = predict(model, predn_samples, predn_timestamps)

        plotter.plot_predictions("SC{} Prediction".format(datasets.END_CYCLE),\
        predn_timestamps, predn_predictions, "SC 24 Prediction.png", compare=True)


if __name__=="__main__":
    is_train = False
    prediction = False
    plotting = False
    scaling = False
    cloudshare = False
    model = (sys.argv[1]).upper()

    main(is_train, prediction, plotting, scaling, model)
    if cloudshare:
        os.system("./cloudshare.sh")
        print(LINESPLIT)
        print("All newly generated files moved to saved outputs and shared with the cloud(Dropbox) successfully!")
