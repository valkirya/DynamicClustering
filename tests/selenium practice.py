# -*- coding: utf-8 -*-
"""
Selenium Script : Logistics Routing Simulation
"""  

# gmailId, passWord = ('lara.cota@mercadolivre.com' , '')
# try: 
#     driver = webdriver.Chrome("C:/Users\lcota\Downloads\chromedriver_win32/chromedriver.exe")
#     driver.get(r'https://accounts.google.com/signin/v2/identifier?continue&flowName=GlifWebSignIn&flowEntry=ServiceLogin') 
#     driver.implicitly_wait(15) 
  
#     loginBox = driver.find_element_by_xpath('//*[@id ="identifierId"]') 
#     loginBox.send_keys(gmailId) 
  
#     nextButton = driver.find_elements_by_xpath('//*[@id ="identifierNext"]') 
#     nextButton[0].click() 
  
#     passWordBox = driver.find_element_by_xpath( 
#         '//*[@id ="password"]/div[1]/div / div[1]/input') 
#     passWordBox.send_keys(passWord) 
  
#     nextButton = driver.find_elements_by_xpath('//*[@id ="passwordNext"]') 
#     nextButton[0].click() 
  
#     print('Login Successful...!!') 
# except: 
#     print('Login Failed') 
    
    
#############################################################################
# 
from selenium import webdriver
import pandas as pd
import os
import time

#  chrome driver 
# download in : https://sites.google.com/a/chromium.org/chromedriver/downloads
driver = webdriver.Chrome("C:/Users\lcota\Downloads\chromedriver_win32/chromedriver.exe")

# Access Logistics 
driver.get("https://envios.mercadolivre.com.br/logistics/planification-simulation")

### Logistics login info
mail, passWord = 'sc.br.lcota','legalonazo70'

loginBox = driver.find_element_by_id('user_id') 
loginBox.send_keys(mail) 
  
nextButton = driver.find_element_by_xpath("//button[@class='andes-button andes-button--large andes-button--loud andes-button--full-width']")
nextButton.click() 

passWordBox = driver.find_element_by_id('password') 
passWordBox.send_keys(passWord) 
  
nextButton = driver.find_element_by_xpath("//button[@class='andes-button andes-button--large andes-button--loud andes-button--full-width']")
nextButton.click()

#####################################################

# Setting SVC work name
SVC = "SMG4"

# attention
# SC_VG
# SC_ZS

placeBox = driver.find_element_by_xpath("//a[@class='sc-label sc-label--selectable']")
placeBox.click()

itemBox = driver.find_elements_by_xpath("//*[contains(text(),'"+SVC+"')]")
for item in itemBox:
   if item.get_attribute("class") == 'list-row__text':
       item.click()
       break

#####################################################

### Getting all .csv file from the data directory

directory_path = r'C:\\Users\\lcota\\Documents\\GitHub\\dynamic-clustering\\data\\outputs\\cenario_cluster_dinamico\\SMG4\\'

src_files = os.listdir(directory_path) 
csv_files = [x for x in src_files if 'csv' in x]

jobIdList = []

#file = 'SMG4_3.csv'
for file in csv_files :
    
    # Simulation start
    newSimulationBox = driver.find_element_by_xpath("//a[@class='andes-button new-button andes-button--small andes-button--loud']")    
    newSimulationBox.click()
    time.sleep(2)
    
    # Do import directly
    #dropBox = driver.find_element_by_xpath("//button[@class='ui-upload__button']")
    #dropBox.click()
    
    # send keys to the input element with type="file" that is responsible for the file upload. 
    file_upload = driver.find_element_by_xpath('//input[@class="ui-upload__input"]')
    file_upload.send_keys(directory_path+file)
    time.sleep(1)
    
    importBox = driver.find_element_by_xpath("//button[@class='andes-button andes-button--large andes-button--loud']")
    importBox.click()
    time.sleep(10)
    
    nextButton = driver.find_element_by_xpath("//a[@class='andes-button andes-button--large andes-button--loud']")    
    nextButton.click()
    time.sleep(2)
    
    # TODO add critérios de roteiro
    nextButton = driver.find_element_by_xpath("//button[@class='andes-button side-content__button andes-button--large andes-button--loud']")
    nextButton.click()
    time.sleep(3)
    
    url = driver.current_url
    jobId = url.rpartition('/')[-1]
    jobIdList.append(jobId)
    
    createButton = driver.find_element_by_xpath("//button[@class='andes-button side-content__button andes-button--large andes-button--loud']")
    createButton.click()
    time.sleep(3)
    
    ######################################################

print(jobIdList)
    
salveRoutingInfo = True
salveRouteInfo = True

routingInfo = {'JobId' : [],
           'NumPacotes' : [],
           'Veículo' : [],
           'Prefixo': [],
           'Rotas': [],
           'DS' : [],
           'Saída':[],
           'SPR' : [],
           'TPR': [],
           'DPR': []
           }

routeInfo = {'JobId' : [],
       'Paradas' : [],
       'Pacotes' : [],
       'Distancia' : [],
       'Duracao' : [],
       'Capacidade' : []
       }
            
## Getting results

if salveRoutingInfo :

    urlRoutings = r'https://envios.mercadolivre.com.br/logistics/planification-simulation/routing/details/'
            
    for jobId in jobIdList :

        # Access url
        urlRoutingDetail = urlRoutings + jobId
        driver.get(urlRoutingDetail)
        
        # routing       
        routingBox = driver.find_element_by_xpath("//p[@class='header-content-text--light']")
        numPackages = int(routingBox.text.rpartition('pacotes')[0])    
        routingInfo['JobId'].append(jobId)
        routingInfo['NumPacotes'].append(numPackages)
        
        routingDetailBox = driver.find_elements_by_class_name("header-content-total-column")
        if len(routingDetailBox) == 8:
            for item in routingDetailBox:
                value = item.text.rpartition('\n')
                routingInfo[value[0]].append(value[2])
        else:
            print("Header elements missing")
          
        if salveRouteInfo : 
                
            routesDetail = driver.find_elements_by_class_name("routing-content__route-list")
                
            rowns = routesDetail[0].text.split('Rota')[2:]
            for item in rowns :
                value = item.split(' ')
                routeInfo['JobId'].append(jobId)
                routeInfo['Paradas'].append( str(value[3]))
                routeInfo['Pacotes'].append( str(value[4]))
                routeInfo['Distancia'].append(str(value[5]))
                routeInfo['Duracao'].append( str(value[7]))
                routeInfo['Capacidade'].append( str(value[9]))  

## Export results
if salveRoutingInfo or salveRouteInfo:
    
    outputPath = r'C:\\Users\\lcota\\Documents\\GitHub\\dynamic-clustering\\tests\\\simulationsOutputs\\'
    outputDirectory = outputPath + SVC + '\\'
    os.makedirs(outputDirectory, exist_ok = True)
    
    routingDf = pd.DataFrame(routingInfo)
    routeDf = pd.DataFrame(routeInfo)
    
    output_file = time.asctime().replace(":","") +'.xlsx'
    writer = pd.ExcelWriter(outputDirectory + output_file)
    routingDf.to_excel(writer, sheet_name='Roteiros', index = False)
    routeDf.to_excel(writer, sheet_name='Rotas', index = False)
    writer.save()
    