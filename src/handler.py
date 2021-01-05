'''
'''

import os
import time
import pandas as pd
import numpy as np
from itertools import compress
from sklearn.preprocessing import StandardScaler

#############################################################################
# Reading parameters from configuration file
###########################################################################

class Parameters ():
    
    def __init__(self, inputs, fileName):
        
        # setting directories
        if fileName is None:
            self.output_directory = '../data//outputs//' 
            self.input_file = '../data//inputs//'
        else:
            asctime = time.asctime().split()
            ext = asctime[2] + asctime[1] + asctime[3].replace(":","-")
            self.output_directory = os.path.dirname(fileName) + '//outputs//' + ext + '//'
            self.input_file = fileName
            
        os.makedirs(self.output_directory, exist_ok=True)
        
        # setting parameters
        self.num_experiments = 10 if int(inputs['GeneralParameters']['num_experiments']) == 0  else int(inputs['GeneralParameters']['num_experiments'])
        self.plot = bool(inputs['GeneralParameters']['plot_maps'])
        
        self.svc_name = str(inputs['ServiceCenterParameters']['name'])
        typ = str(inputs['ServiceCenterParameters']['type'])
        
        if typ == "default":
            default_list = inputs['DefaultAssigment']
            self.type = default_list[self.svc_name] if self.svc_name in default_list.keys() else "kmeans"
        elif typ in ("ward", "kmeans"):
            self.type = typ
        else:
            self.type = "kmeans"
        
        self.cluster_max_size = 1500 if int(inputs['RoutingParameters']['cluster_max_size']) == 0 else int(inputs['RoutingParameters']['cluster_max_size'])
        self.cluster_min_size = int(inputs['RoutingParameters']['cluster_min_size'])
        
        self.vol_filter_lower_bound = float(inputs['RoutingParameters']['lower_bound_vol_filter'])
        self.vol_filter_upper_bound = 1e5 if float(inputs['RoutingParameters']['upper_bound_vol_filter']) == 0 else float(inputs['RoutingParameters']['upper_bound_vol_filter'])        
        self.vol_filter_min_cluster_size = float(inputs['RoutingParameters']['minimum_cluster_size_vol_filter'])
        
        self.dbscan_maximum_distance = float(inputs['DBSCANParameters']['maximum_distance'])
        self.dbscan_neighborhood_size = float(inputs['DBSCANParameters']['neighborhood_size'])
        
        self.kmeans_tolerance = float(inputs['KmeansParameters']['tolerance'])
        self.kmeans_num_initialization = int(inputs['KmeansParameters']['num_initialization'])
        
        self.ward_num_neighbors = int(inputs['HierarchicalParameters']['num_neighbors'])
        
        self.balance_nun_closest_neighbors = int(inputs['BalanceParameters']['num_closest_neighbors'])
        self.balance_feasibility_threshold = int(inputs['BalanceParameters']['feasibility_threshold'])
        
        self.osrm_request_limit = int(inputs['RefiningParameters']['osrm_request_limit'])
        self.neighborhood_size = int(inputs['RefiningParameters']['neighborhood_size'])


#############################################################################

##########################################################################
               
class PreProcessing(): 
              
    def read_input_data (file_directory):
        data = pd.read_csv(file_directory, error_bad_lines=False)
        if len(data.columns) == 1: data = pd.read_csv(file_directory, sep = ";")
    
        return data
    
    def split_input_data (data, lower_bound, upper_bound, min_size):
        if 'Volume' in data.columns or 'volume' in data.columns:
            index = data.filter(regex='olume').columns
            volume_values = data[index].values
            index_first_block = [i for i, n in enumerate(volume_values) if float(n) <= upper_bound]
            index_last_block = [i for i, n in enumerate(volume_values) if float(n) >= lower_bound]
            
            if len(index_first_block) < min_size : index_first_block = []
            if len(index_last_block) < min_size : index_last_block = []
    
            index_middle_block = list(set(range(len(volume_values))) - set(index_first_block)  - set(index_last_block))
            
            data_union = {"_vol_upper_" + str(upper_bound):index_first_block,
                          "_vol_lower_" + str(lower_bound):index_last_block,
                          "" : index_middle_block}
            
            data_repartition = {key:value for (key,value) in data_union.items() if len(value)>0 }
            
        else:
            data_repartition = {"" : list(range(len(data)))}
        
        return data_repartition
    
    def validate_input_data (data):
        
        message = ""
        index = data.filter(regex='olume').columns
        if data[index].isna().values.any():
            values = np.unique(np.where(data[index].isna())[0])
            data.drop(values, inplace = True)
            data.reset_index(drop=True, inplace=True)   
            message = message + "Coluna volume contém valores vazios nas posições {}".format(values)
       
        index = data.filter(regex='hipm').columns
        if data[index].isna().values.any():
            values = np.unique(np.where(data[index].isna())[0])
            data.drop(values, inplace = True)
            data.reset_index(drop=True, inplace=True)            
            message = message + "Coluna Shipment contém valores vazios nas posições {}".format(values)
            
        index = data.filter(regex='atit|ongi').columns
        if data[index].isna().values.any():
            values = np.unique(np.where(data[index].isna())[0])
            data.drop(values, inplace = True)
            data.reset_index(drop=True, inplace=True)   
            message = message + "Lat Long contém valores vazios nas posições {}".format(values)
        
        alert = "Input data format has been validated!" if message == "" else message
        
        return (data, alert)
        
    def filter_input_data (data):
        #Just for test
        # data = data.copy()
        # data.Correcao = data.Correcao.astype(bool)
        # data = data[~data.Correcao]
        # data = data.loc[data['Route_ID'].notnull()]
        # data = data.reset_index()
        
        index = data.filter(regex='hipm|atit|ongi').columns
        data_filtered = data[index]
        data_filtered.columns = ["Shipment", 'Lat', 'Long']
        
        return data_filtered
            
    def get_geolocate_data (data):
        data_filter = data[['Lat','Long']]
        
        return data_filter
    
    def get_scaled_data (data):
        data_df = pd.DataFrame(StandardScaler().fit_transform(data[['Lat','Long']]))

        return data_df

#############################################################################

###########################################################################

class PostProcessing():
    
    # update cluster labels if same geolocation is in more than one cluster
    def validate_output_data (data_df, cluster_labels, cluster_sizes ):
        
        labels = cluster_labels.copy()
        sizes = pd.Series(cluster_sizes) 
        
        geolocation_data = {"Key" :  [i + j for i, j in zip(data_df["Lat"].astype(str), data_df["Long"].astype(str))], "Cluster" : labels}
        df = pd.DataFrame(geolocation_data)
        viol_mask = df.groupby('Key').Cluster.nunique() > 1
        violated_geolocation_points = viol_mask[viol_mask].index.tolist()
        
        for x in violated_geolocation_points:
            filtered = df.Cluster[df.Key==x]
            ideal_cls = sizes[filtered.values].idxmin() 
            labels[filtered.index] = ideal_cls
         
        violated = list(compress(range(len(viol_mask)), viol_mask))
        sizes = pd.Series(labels).value_counts()
        
        return (labels, sizes, violated)
            
    def generate_output_data (data_df, cluster_labels):

        output_data = {"Shipment" : data_df['Shipment'] , "Cluster" : cluster_labels}
        
        return pd.DataFrame(output_data)
    
    def generate_output_folder (output_directory, svc_name, n_packages, key):
        
        folder_name = str(svc_name) + '_' + str(n_packages) + str(key)
        folder_directory = output_directory + folder_name 
        os.makedirs(folder_directory, exist_ok=True)
        
        return folder_directory, folder_name
        
    def write_output_data (output_data, folder_directory):
        
        clusters_names = output_data.Cluster.unique()
        
        for n in clusters_names:
            filter_data = output_data.Shipment[output_data.Cluster == n]
            output_file_name = str(n) +"_"+ str(len(filter_data))
            filter_data.to_csv( folder_directory + "/" + output_file_name +".csv", index=False, header = False)
            
    def write_cluster_map (cluster_map, num_cluster, folder_name, folder_directory):
               
        file_name = folder_name + '_' + str(num_cluster)
        cluster_map.save(folder_directory + "/" + file_name +'.html')  
         
        