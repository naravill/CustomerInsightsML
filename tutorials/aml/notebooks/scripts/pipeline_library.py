# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import tempfile
from azureml.core import Dataset, Run, Datastore
import pandas as pd
import numpy as np
import sys
import joblib
from azureml.core.model import Model
from azureml.core import Workspace
from sklearn.linear_model import LogisticRegression
import datetime

def pipeline_steps(hotelActivityData, customerData, serviceUsageData, config):
    #variables from config file
    output_datastore = config["output_datastore"]
    output_path = config["output_path"]
    model = config["model"]
    run = config["run"] 
    step_type = config["step_type"]
    workspace = config["workspace"]

    hotelActivityColumns = [ "HotelCustomerID","CheckInDate","CheckOutDate","RoomType","DollarsSpent","BookingType","TravelCategory","NumberOfNights"]
    hotelActivityDF = hotelActivityData.to_pandas_dataframe()[hotelActivityColumns]

    # select CustomerId,ContosoHotel_HotelCustomers_HotelCustomerID from customer data
    customerDataColumns = [ "CustomerId","ContosoHotel_HotelCustomers_HotelCustomerID"]
    customerDataDF = customerData.to_pandas_dataframe()[customerDataColumns]

    # select ServicesCustomerID,ServiceName,ServiceDate,ServiceCost from Service Usage data
    serviceUsageDataColumns = [ "ServicesCustomerID", "ServiceName", "ServiceDate", "ServiceCost"]
    serviceUsageDF = serviceUsageData.to_pandas_dataframe()[serviceUsageDataColumns]

    # join hotel stay activity data on HotelCustomerID and customer data on ContosoHotel_HotelCustomers_HotelCustomerID inner join
    df_joined1 = hotelActivityDF.merge(customerDataDF, left_on="HotelCustomerID", right_on="ContosoHotel_HotelCustomers_HotelCustomerID")
    # select columns on joined dataset HotelCustomerID,RoomType,DollarsSpent,BookingType,TravelCategory,CheckInDate,CheckOutDate,NumberOfNights,CustomerId
    df_joined1_selected = df_joined1[["HotelCustomerID","RoomType","DollarsSpent","BookingType","TravelCategory","CheckInDate","CheckOutDate","NumberOfNights","CustomerId"]]
    # sql transformation on previous dataframe 
    df_joined1_selected_after_sql_transform = df_joined1_selected[df_joined1_selected["CheckInDate"]<"2016-12-31T00:00:00"]

    # change RoomType,BookingType,TravelCategory from previous dataframe to categorical type and features
    df_after_categorical_type_change = df_joined1_selected_after_sql_transform
    columns_for_categorical_type_change = ["RoomType","BookingType","TravelCategory"]
    for col in columns_for_categorical_type_change:
        df_after_categorical_type_change[col] = df_after_categorical_type_change[col].astype('category')

    # change to indicator values type RoomType,BookingType,TravelCategory from previous dataframe
    df_after_change_to_indicator_values = pd.get_dummies(df_after_categorical_type_change, columns = columns_for_categorical_type_change)
    
    # change datatype to RoomType-Large,RoomType-Small,BookingType-Online,BookingType-Phone Call,TravelCategory-Business,TravelCategory-Leisure to integer
    columns_to_convert_to_integer = ["RoomType_Large","RoomType_Small","BookingType_Online","BookingType_Phone Call","TravelCategory_Business","TravelCategory_Leisure"]
    for col in columns_to_convert_to_integer:
        df_after_change_to_indicator_values[col] = df_after_change_to_indicator_values[col].astype(int)
    
    groupby_columns = columns_to_convert_to_integer + ["DollarsSpent","HotelCustomerID"]
    # change names after groupby
    column_names_after_groupby = ["RoomTypeLargeCount","RoomTypeSmallCount","BookingTypeOnlineCount","BookingTypePhoneCallCount","TravelCategoryBusinessCount","TravelCategoryLeisureCount","TotalDollarSpent"]
    df1 = df_after_change_to_indicator_values[groupby_columns].groupby(by=["HotelCustomerID"]).sum()
    df1.rename(columns = {"RoomType_Large":"RoomTypeLargeCount", 
    "RoomType_Small":"RoomTypeSmallCount", 
    "BookingType_Online":"BookingTypeOnlineCount", 
    "BookingType_Phone Call":"BookingTypePhoneCallCount",
    "TravelCategory_Business":"TravelCategoryBusinessCount",
    "TravelCategory_Leisure":"TravelCategoryLeisureCount",
    "TravelCategory_Leisure":"TravelCategoryLeisureCount",
    "DollarsSpent":"DollarsSpentCount"}, inplace = True) 
    df2 = df_after_change_to_indicator_values[["HotelCustomerID","CustomerId"]].drop_duplicates()
    df_left = df1.merge(df2,left_on = "HotelCustomerID",right_on = "HotelCustomerID")
    
    # convert NumberOfNights from the dataframe after join and select columns to integer
    df_joined1_selected["NumberOfNights"] = df_joined1_selected["NumberOfNights"].astype(int)
    # apply sql transformation output is dfright
    def transform_dataframe_column(df,mask,col_name, col_from_dataframe ="NumberOfNights"):
        df[col_name] = df[col_from_dataframe]
        df[col_name][~mask] = 0
        return df

    def transform_dataframe_staycount_columns(df,mask,col_name):
        df[col_name] = np.zeros(df.shape[0])
        df[col_name][mask] = 1
        return df
        
    df_before_sql_transform = df_joined1_selected

    mask = df_before_sql_transform["CheckOutDate"]<="2016-12-31T00:00:00"
    df_before_sql_transform = transform_dataframe_column(df_before_sql_transform, mask, col_name = "StayDayCount", col_from_dataframe = "NumberOfNights")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2016-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2015-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_column(df_before_sql_transform, mask, col_name = "StayDayCount2016", col_from_dataframe = "NumberOfNights")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2015-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2014-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_column(df_before_sql_transform, mask, col_name = "StayDayCount2015", col_from_dataframe = "NumberOfNights")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2014-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2013-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_column(df_before_sql_transform, mask, col_name = "StayDayCount2014", col_from_dataframe = "NumberOfNights")


    mask = df_before_sql_transform["CheckOutDate"] <= "2016-12-31T00:00:00" 
    df_before_sql_transform = transform_dataframe_staycount_columns(df_before_sql_transform, mask, col_name = "StayCount")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2016-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2015-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_staycount_columns(df_before_sql_transform, mask, col_name = "StayCount2016")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2015-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2014-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_staycount_columns(df_before_sql_transform, mask, col_name = "StayCount2015")

    mask = (df_before_sql_transform["CheckOutDate"] <= "2014-12-31T00:00:00") & (df_before_sql_transform["CheckOutDate"] >= "2013-01-01T00:00:00")
    df_before_sql_transform = transform_dataframe_staycount_columns(df_before_sql_transform, mask, col_name = "StayCount2014")

    columns_sum = ["StayDayCount","StayDayCount2016", "StayDayCount2015", "StayDayCount2014", "StayCount", "StayCount2016", "StayCount2015", "StayCount2014"]
    df_unique_hotelcustomerID = df_before_sql_transform[["HotelCustomerID"]].drop_duplicates().set_index("HotelCustomerID")
    df_unique_hotelcustomerID[columns_sum] = df_before_sql_transform.groupby(by = "HotelCustomerID").sum()[columns_sum]
    df_unique_hotelcustomerID[["FirstStay"]] = df_before_sql_transform.groupby(by = "HotelCustomerID").min()[["CheckInDate"]]
    df_unique_hotelcustomerID[["LastStay"]] = df_before_sql_transform.groupby(by = "HotelCustomerID").max()[["CheckInDate"]]
    stay_info = df_unique_hotelcustomerID[columns_sum+["FirstStay","LastStay"]].reset_index()

    # second part of the SQL query
    df_right = stay_info[["HotelCustomerID","StayDayCount","StayDayCount2016","StayDayCount2015","StayDayCount2014","StayCount","StayCount2016","StayCount2015","StayCount2014","FirstStay","LastStay"]]
    df_right["UsageTenure"] = pd.to_datetime("2016-12-31T00:00:00")-stay_info["FirstStay"]
    df_right["Label"] = np.ones(df_right.shape[0])
    df_right["Label"][df_right["LastStay"] > "2016-12-31T00:00:00"] = 0
    
    df_left = df_left.merge(df_right,left_on = "HotelCustomerID",right_on  ="HotelCustomerID")
    
    def transform_serviceusage(df,col_name,serviceUsageDF,mask_col_name):
        mask = serviceUsageDF["ServiceName"] == mask_col_name
        df[col_name] = np.zeros(df.shape[0])
        df[col_name] = serviceUsageDF[mask]["ServiceCost"]
        return df


    df_right = serviceUsageDF[["ServicesCustomerID"]]
    df_right["ServiceDate"] = serviceUsageDF["ServiceDate"]
    df_right = transform_serviceusage(df = df_right, col_name = "ConciergeUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "concierge")
    df_right = transform_serviceusage(df = df_right, col_name = "CourierUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "courier")
    df_right = transform_serviceusage(df = df_right, col_name = "DryCleaningUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "dry_cleaning")
    df_right = transform_serviceusage(df = df_right, col_name = "GymUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "gym")
    df_right = transform_serviceusage(df = df_right, col_name = "PhoneUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "phone")
    df_right = transform_serviceusage(df = df_right, col_name = "RestaurantUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "restaurant")
    df_right = transform_serviceusage(df = df_right, col_name = "SpaUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "spa")
    df_right = transform_serviceusage(df = df_right, col_name = "TelevisionUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "television")
    df_right = transform_serviceusage(df = df_right, col_name = "WifiUsage", serviceUsageDF = serviceUsageDF,mask_col_name = "wifi")
    groupby_columns = ["ConciergeUsage","CourierUsage","DryCleaningUsage","GymUsage","PhoneUsage","RestaurantUsage","SpaUsage","TelevisionUsage","WifiUsage","ServicesCustomerID"]
    df_right = df_right[df_right["ServiceDate"] < "2016-12-31T00:00:00"][groupby_columns].groupby(by=["ServicesCustomerID"]).sum()
    df_right = df_right.reset_index()

    df_joined = df_left.merge(df_right, left_on = "HotelCustomerID", right_on = "ServicesCustomerID")

    df_joined["UsageTenure"] = df_joined["UsageTenure"].fillna(pd.Timedelta(seconds=0))


    df_joined["UsageTenure"] = df_joined["UsageTenure"].astype(int)/(24*60*60*1e9)
    
    cols_labels = [col for col in df_joined.columns if col not in ["FirstStay","LastStay"]]
    df_joined = df_joined[cols_labels]
    df_joined.fillna(0,inplace = True)

    if step_type =="test":
        df_result = df_joined
        cols = [col for col in df_joined.columns if col not in ["Label","HotelCustomerID","CustomerId","ServicesCustomerID"]]
        df_result = write_results(df_result, cols, output_datastore, output_path, model,run)
    elif step_type == "train":
        permuted_indices = np.random.permutation(df_joined.index)
        train_len = int(0.8*len(permuted_indices))
        train = df_joined.iloc[permuted_indices[:train_len]]
        test = df_joined.iloc[permuted_indices[train_len:]]
        cols = [col for col in train.columns if col not in ["Label","FirstStay","LastStay","HotelCustomerID","CustomerId","ServicesCustomerID"]]
        
        print("train columns")
        model_folder = config["model_folder"]
        model_name = config["model_name"]
        train_steps(train, test, cols, model_folder, model_name)

        print("register model")
        model_path = "./" + config["model_folder"] 
        description = config["description"]
        register_model(model_path, model_name, description, workspace)
    else:
          raise Exception("Invalid step type, allowed values are train and test")
    return 

def train_steps(train, test, cols, model_folder, model_name):
    print("training ...")
    clf = LogisticRegression()
    clf.fit(train[cols].values, train["Label"].values)

    print('predicting ...')
    y_hat = clf.predict(test[cols].astype(int).values)

    acc = np.average(y_hat == test["Label"].values)
    print('Accuracy is', acc)

    print("save model")
    os.makedirs('models', exist_ok=True)    
    joblib.dump(value=clf, filename= model_folder +'/'+ model_name + ".pkl")
    return

def register_model(model_path, model_name, description, ws):
    model = Model.register(model_path = model_path,
                        model_name = model_name,
                        description = description,
                        workspace = ws)
    return 
    
def write_results(df, cols, output_datastore, output_path, model, run):

    ws = run.experiment.workspace
    datastore = Datastore.get(ws, output_datastore)
    output_folder = tempfile.TemporaryDirectory(dir = "/tmp")
    filename = os.path.join(output_folder.name, os.path.basename(output_path))
    print("Output filename: {}".format(filename))

    try:
        os.remove(filename)
    except OSError:
        pass
    
    df["ScoredLabels"] = model.predict(df[cols].astype(int).values)
    print("resultLabels", df["ScoredLabels"].iloc[:10])
    df["ScoredProbabilities"] = model.predict_proba(df[cols].astype(int).values)[:,1]
    print("resultProbabilities", df["ScoredProbabilities"].iloc[:10])

    # set HotelCustomerID to index to remove the column1 columns in the dataframe
    df = df.set_index("CustomerId")

    directory_name =  os.path.dirname(output_path)
    print("Extracting Directory {} from path {}".format(directory_name, output_path))
    
    df.to_csv(filename)
    
    # Datastore.upload() is supported currently, but is being deprecated by Dataset.File.upload_directory()
    # datastore.upload(src_dir=output_folder.name, target_path=directory_name, overwrite=False, show_progress=True)
    # upload_directory can fail sometimes.
    output_dataset = Dataset.File.upload_directory(src_dir=output_folder.name, target = (datastore, directory_name))
    return df