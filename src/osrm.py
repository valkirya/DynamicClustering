
"""
"""

import requests
import json

def call_request (data, num_cluster):
    
    url = 'http://router.project-osrm.org/table/v1/driving/'
    
    # define data, destination and source index 
    waypoints = ';'.join(map(lambda pt: '{},{}'.format(*reversed(pt)), data))
    dest = ';'.join(map(lambda pt: '{}'.format(pt), range(num_cluster)))
    src = ';'.join(map(lambda pt: '{}'.format(pt), range(num_cluster, len(data))))
        
    # request accepts maximum of 8192 bytes  
    try:
        r = requests.get('{}{}?sources={}&destinations={}&annotations=duration,distance'.format(url, waypoints, src, dest))
        dictContent = json.loads(r.content.decode('utf-8'))
        distance = dictContent['distances']
        duration = dictContent['durations']
        
    except Exception as e :
        print(e)
        
    return (distance, duration)
        

    

        
        

            

        
        
        