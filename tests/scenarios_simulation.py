# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:11:12 2020

@author: lcota
"""

import pandas as pd
import os

DATA_DIRECTORY = 'data/inputs'
OUTPUT_DIRECTORY = 'data/outputs/cenario_operação'

file_name_list =  [
 'SAM1.csv',
 'SBA1.csv',
 'SDF1.csv',
 'SGO1.csv',
 'SMG4.csv',
 'SRJ1.csv',
 'SRS1AM.csv',
 'SRS1PM.csv',
 'SSP15.csv',
 'SSP4.csv',
 'SSP7AM.csv',
 'SSP7PM.csv',]

file_name_list = ['SQR1.csv', 'STL1.csv']

for f in file_name_list:
    print(f)
    
    data = pd.read_csv("../" + DATA_DIRECTORY + "/" + f, sep = ";")
    data.Correcao = data.Correcao.astype(bool)
    data = data[~data.Correcao]
    data = data.loc[data['Order_ID'].notnull()]
    data = data.reset_index()
    print(len(data))
    roteiros_id = data.Order_ID.unique()
    for r in roteiros_id:
        filter_data = data.Shipment[data.Order_ID == r].astype(str)
        file_name_out = os.path.splitext(f)[0] + "_"+ str(int(r))
        directory_path = "../" + OUTPUT_DIRECTORY+ "/" + str(os.path.splitext(f)[0])
        os.makedirs(directory_path, exist_ok=True)    
        filter_data.to_csv( directory_path + "/" +file_name_out +".csv", index=False, header = False)
    
