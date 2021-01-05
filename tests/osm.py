# -*- coding: utf-8 -*-
"""
 pip install osrm-py

"""

import asyncio
import aiohttp
import osrm

loop = asyncio.get_event_loop()

async def request():
    client = osrm.AioHTTPClient(host='http://localhost:5000')
    response = await client.table(
        coordinates=[[-74.0056, 40.6197], [-74.0034, 40.6333]],
        overview=osrm.overview.full)
    print(response)
    await client.close()

loop.run_until_complete(request())
    
#client = osrm.AioHTTPClient(host='http://localhost:5000')
#table_response = client.table.get(sources=[(-6.5962986, 106.7972421)], #destinations=[(-6.17126, 106.64404)])

#host = 'http://localhost:5000/'
#path = 'table/v1/driving/'

import requests
import json
from math import ceil
import osrm

url = 'http://router.project-osrm.org/table/v1/driving/'


centers = [[19.56059709615385, -99.76336294230768],
 [19.253878745185183, -99.60281324814815],
 [19.189837352941176, -100.12784732679738],
 [19.287233487082545, -99.67881726780088],
 [19.957028212389382, -99.53915569911504],
 [18.904062202898547, -100.14840332850243],
 [19.272795221917807, -99.49985387397261],
 [19.79516434653465, -99.87472724752477],
 [19.017475946859904, -99.58564177294687],
 [19.331286211586903, -99.59625264483627]]

# all data =  centers + data points
#waypoints = [(-30.59464,-71.19366),(-30.60386,-71.21342),(-30.58654,-71.18429),(-30.60298,-71.20075),(-30.58603,-71.19196)]
#waypoints = ';'.join(map(lambda pt: '{},{}'.format(*reversed(pt)), waypoints))

# convert list of list in list of tuples
centers_data = list(map(tuple, centers)) 
points_data = list(map(tuple, geo_data.to_numpy())) 

total_points = len(points_data)
osmRequestLimit = 1000
nPointsPerRequest = ceil(osmRequestLimit/num_cluster)
nRequests = int(total_points/nPointsPerRequest)
nPointsLastRequest = total_points-(nRequests*nPointsPerRequest)

matrix_centers_dist = []

# Create Function to return src and dest
# input centers data , points data , destination , source 

#all_data = centers_data + points_data
all_data = centers_data + points_data[:nPointsPerRequest]
waypoints = ';'.join(map(lambda pt: '{},{}'.format(*reversed(pt)), all_data))

# cluster centers index
destination = list(range(num_cluster))
dest = ';'.join(map(lambda pt: '{}'.format(pt), destination))

# deliveries points index
start = num_cluster
limit = nPointsPerRequest
end = num_cluster + limit #len(geo_data)
source = list(range(start,end))
src = ';'.join(map(lambda pt: '{}'.format(pt), source))

# request a maximum of 8192 bytes 

try:
    to = time.time()
    r = requests.get('{}{}?sources={}&destinations={}&annotations=distance'.format(url, waypoints, src, dest))
    print(time.time()-to)
except Exception:
    print (Exception)
    
if r.status_code == 200:

    #strContent = r.content.decode(encoding='UTF-8')
    dictContent = json.loads(r.content.decode('utf-8'))
    data = dictContent['distances']
    matrix_centers_dist.extend(data)
    
else:
    return 0


centers_dist = np.array(matrix_centers_dist)

