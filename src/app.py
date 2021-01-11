#************************************************************
# Clustering Geospatial Data - Dynamic Clustering
# Author: Julia Couto and Lara Cota
# Mercado Envios
#************************************************************

# Import libraries
from src.handler import Parameters, PreProcessing, PostProcessing
from src.model import Clusters
from src.measures import Measure
#from src.visualization import Plots
from src.logInfo import AppLogging 

import json
import os, sys

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Main Function 
def run_model (inputs, file_name = None):  

    with open(resource_path('data/configuration.json'), encoding = 'UTF8') as input_json:
        configs = json.load(input_json)
   
    ####################### Reading Parameters ######################## 
    
    param = Parameters (configs, inputs, file_name)   
    AppLogging.setConfig(param.output_directory)
    AppLogging.startMessage(param)
           
    ####################### Reading Input Data ########################   
    
    input_data = PreProcessing.read_input_data (param)
    data, alerts = PreProcessing.filter_input_data(input_data) 
    AppLogging.inputReadingMessage (alerts)
    
    data, alerts = PreProcessing.validate_input_data (data)
    AppLogging.inputValidationMessage (alerts)
    
    data_repartition = PreProcessing.split_input_data (data, param.vol_filter_lower_bound, param.vol_filter_upper_bound, param.vol_filter_min_cluster_size)
    AppLogging.volumeFilteringMessage(data_repartition, param)
    
    try:
        for (key,value) in data_repartition.items():
            
            AppLogging.dataMessage(key, value)
            filter_data = data.iloc[value].reset_index(drop=True)

            ####################### Pre Processing Data ########################
                        
            geo_data = PreProcessing.get_geolocate_data (filter_data)
            model_data = PreProcessing.get_scaled_data (geo_data)
      
            ############### Choosing the Appropriate Number of Clusters #############
           
            min_num_cluster = Measure.calculate_min_num_cluster (model_data, param)
            num_cluster, num_tests = Measure.calculate_best_num_cluster (model_data, param, min_num_cluster)
            AppLogging.metricsMessage(min_num_cluster, num_tests, num_cluster)

            ###################  1) Call Clustering Algorithm ####################
            
            model = Clusters(model_data, geo_data, num_cluster, param)
            labels, sizes, algo_name = model.clustering_algorithm ()
            AppLogging.algorithmNameMessage(algo_name)
            
            ################# 2.1) Call Cluster Aglumerative Algorithm  ################# 
            
            violation = model.check_aglomerative_violation(sizes)
            
            if violation:
                labels, sizes, clusters_viol = model.aglomerative_clusters(labels, sizes) 
                AppLogging.aglomerativeMessage(clusters_viol)
            
            AppLogging.numClustersMessage(model.num_cluster) 
   
            ################# 2.2) Call Cluster Balance Algorithm  #################  
            
            violation = model.check_balance_violation(sizes)
            
            if violation:
                
                labels, sizes, clusters_viol = model.balance_clusters(labels, sizes) 
                if param.type == 'dense': labels, sizes = model.simple_center_adjustment(labels, sizes)
                AppLogging.balanceMessage(clusters_viol)

            ################# 3) Call Fine Adjustment   ########################
                
            labels, sizes = model.fine_adjustement (labels, sizes)         
            
            #################  Post Processing Data   ########################
           
            labels, sizes, violated_points = PostProcessing.validate_output_data (filter_data, labels, sizes) 
            AppLogging.latlongAdjustmentMessage(violated_points) 
            
            output_data = PostProcessing.generate_output_data (filter_data, labels)
            output_folder, output_name = PostProcessing.generate_output_folder (param.output_directory, param.svc_name, len(output_data), key)
            PostProcessing.write_output_data(output_data, output_folder)

            #################  Print Maps   ########################
            
            if param.plot :
                print()
                #centers = list(map( lambda x : model.calculate_geo_cluster_center(labels, x), range(model.num_cluster)))
                #cluster_map = Plots.cluster_iteractive_view(model.geo_data, labels, sizes, centers)  
                #PostProcessing.write_cluster_map (cluster_map, model.num_cluster, output_name, output_folder)
                     
            AppLogging.endMessage(sizes, output_folder)  
            
        return (True, param.output_directory)
    
    except Exception as e:
        AppLogging.errorMessage()         
        return (False, e)
               