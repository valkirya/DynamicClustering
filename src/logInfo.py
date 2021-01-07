import logging

class AppLogging ():
        
    def setConfig(output_directory, logLevel = logging.INFO):

        output_file = output_directory + 'loginfo.log'
        logging.basicConfig(filename = output_file, filemode= 'w', force = True, level=logLevel, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
                
    def startMessage(param):
        logging.info('Clustering Dynamic in execution with following settings:\nService Center:{}, Type:{}, Cluster max size:{}, Cluster min size:{}'.format(param.svc_name, param.type, param.cluster_max_size, param.cluster_min_size))
    
    def metricsMessage (min_num_cluster, num_tests, num_cluster):
        logging.info('Metrics:\nminimum number of cluster: {}, number of tests: {}, initial number of cluster: {}'.format(str(min_num_cluster), str(num_tests) , str(num_cluster)) )
        
    def inputValidationMessage (alerts): logging.warning(alerts)
             
    def volumeFilteringMessage (data_repartition, param):
        if len(data_repartition) > 1 :
            logging.info('Volume Filtering: < {}, > {}, ClusterMin Size: {}'.format(param.vol_filter_lower_bound,param.vol_filter_upper_bound, param.vol_filter_min_cluster_size))
            
    def dataMessage (key, value):
        logging.info('\n')
        logging.info('Running filtered data: {}, Data size: {}'.format("remain" if str(key) == "" else str(key), str(len(value))) )
            
    def algorithmNameMessage (algo):    
        logging.info('Clustering algorithm used: {}'.format(algo))
        
    def numClustersMessage (num_cluster):    
        logging.info('Number of cluster: {}'.format(num_cluster))
    
    def aglomerativeMessage (clustersNames):    
        logging.info('Call Aglomerative algorithm for clusters of sizes:\n{}'.format(clustersNames))
        
    def balanceMessage (clustersNames):    
        logging.info('Call Balance algorithm for clusters: {}'.format(clustersNames))
        
    def latlongAdjustmentMessage (points):
        if len(points) > 0: logging.info('Clusters adjustement due to lat long in: {} '.format(points))
        
    def endMessage (sizes, folder_directory):    
        logging.info('Clustering results saved in {}.\nClusters created: \nName  Size\n{}'.format(folder_directory, sizes))
        logging.info('\n')
        
    def errorMessage ():
        logging.error('The algorithm returned an Error!', exc_info=True)
    
    def statusMessage ():
        logging.info('n')
        
                
        