# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
from azureml.core import Dataset, Datastore, Run, Workspace
from azureml.data import OutputFileDatasetConfig
from azureml.core.model import Model
import pandas as pd
import numpy as np
import sys
import joblib

import pipeline_library as pl

def score_pipeline(hotelActivityData, customerData, serviceUsageData, config):
    print("scoring ...")
    pl.pipeline_steps(hotelActivityData, customerData, serviceUsageData, config)
    return 

parser = argparse.ArgumentParser("score")

parser.add_argument("--input_data1", type=str, help="data 1")
parser.add_argument("--input_data2", type=str, help="data 2")
parser.add_argument("--input_data3", type=str, help="data 3")
parser.add_argument('--output_path', dest='output_path', required=True)
parser.add_argument('--output_datastore', dest='output_datastore', required=True)

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

print("geting datasets ...")
hotel_activity_data = Run.get_context().input_datasets['HotelStayActivity_dataset']
customer = Run.get_context().input_datasets['Customer_dataset']
service_usage = Run.get_context().input_datasets['ServiceUsage_dataset']

print("Output Location", args.output_datastore + args.output_path)

# Load Model 
print("loading model ...")
model_path = Model.get_model_path("churnscore", _workspace = ws,version = 1)
print("model_path : ", model_path)
model = joblib.load(model_path)


config = {
    "output_datastore" : args.output_datastore,
    "output_path" : args.output_path,
    "model" : model,
    "run" : run,
    "workspace": ws,
    "step_type" : "test"
}

score_pipeline(hotel_activity_data, customer, service_usage, config)