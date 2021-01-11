import PySimpleGUI as sg
#import PySimpleGUIWeb as sg
import json
import numpy as np
from src.app import run_model
       
def createGUI ():
    
    #Define GUI theme
    sg.theme('LightGrey1')
    sg.set_options(auto_size_buttons=True)
        
    #Define window layout
    layout = [
                [
                    sg.Text('Clusterização Dinâmica ', font=('Helvetica', 13), justification='center', size=(60, 1)),
                 ],
                [
                    sg.Text(""),
                ],
                [
                    sg.Text("Selecione o arquivo de dados: "),
                    sg.In(size=(35, 1), enable_events=True, key="-DATA-"),
                    sg.FileBrowse(file_types=(("CSV File", "*.csv"),),
                                  button_text="Selecionar"),
                ],
                # [
                #     sg.Text("Selecione o arquivo de configuração: "),
                #     sg.In(size=(35, 1), enable_events=True, key="-CONFIG-"),
                #     sg.FileBrowse(file_types=(("JSON File", "*.json"),),
                #                   button_text="Selecionar"),
                # ],
                #[sg.Text(""),],
                [
                    sg.Text('_'  * 100, size=(65, 1))
                ],
                [
                    sg.Text('Configuração', font=('Helvetica', 11), justification='left', size=(25, 1)),
                    sg.Text('Filtros de Volumetria', font=('Helvetica', 11), justification='left', size=(30, 1))
                ],
                [   sg.Listbox(["Auto", "Denso", "Semi-Denso", "Disperso"], size=(20,4), enable_events=True, key='-LIST-', default_values= "Auto"),
                     sg.Text("", size=(7,4)),  
                     sg.Text('Até', size=(5, 1)),
                     sg.Spin(values=[i for i in np.array(range(0, 100, 1))/1000], initial_value=0, size=(7, 1), key = '-UPPER-'),
                     sg.Text('A partir de', size=(8, 1)),
                     sg.Spin(values=[i for i in np.array(range(0, 100, 1))/1000], initial_value=0, size=(7, 1), key = '-LOWER-'),
                     #sg.Text('Qtde minima de pacotes', size=(18, 1)),
                     #sg.In(default_text=50, size=(7, 1))],                     
                ],
                [
                    sg.Text(""),   
                ],                
                [
                    sg.Button("Executar", enable_events=True, key="-RUN-" ),
                ],
                [
                    sg.Text(""),  
                ],
                [
                    sg.Text(size=(68,2), key='-OUTPUT-', text_color='black', background_color='whitesmoke')
                ],
            ]
        
    # Create the window
    window = sg.Window("Dynamic Clustering 0.1.0", layout, margins=(50, 20))
        
    # Create GUI variables
    fileName = ''
    inputs = {}
            
    # Create an event loop
    while True:
            
        try:
            
            event, values = window.read()
                
            # End program if user closes window or presses the OK button
            if event == sg.WIN_CLOSED:
                break
                
            # Folder name was filled in, make a list of files in the folder
            if event == "-DATA-":               
                fileName = values["-DATA-"]

            if event == "-CONFIG-":
                configFileName = values["-CONFIG-"]
                with open(configFileName, encoding = 'UTF8') as input_json:
                    inputs = json.load(input_json)                
                        
            if event == "-RUN-":
                                
                if fileName : #and any(inputs):
                    
                    inputs = {}
                    inputs["RoutingParameters"] = {}
                    inputs["ServiceCenterParameters"] = {}
                    inputs["RoutingParameters"]["lower_bound_vol_filter"] = float(values["-LOWER-"])
                    inputs["RoutingParameters"]["upper_bound_vol_filter"] = float(values["-UPPER-"])
                    inputs["ServiceCenterParameters"]["type"] = str(values["-LIST-"][0])
                    
                    window['-RUN-'].update(disabled=True)
                    window['-OUTPUT-'].update("Clusterização Dinâmica em execução ...")       
                    window.Refresh()
                    
                    status, output = run_model (inputs, fileName)
                    window['-RUN-'].update(disabled=False)
                    
                    if status:
                        message = 'Execução Finalizada! Arquivos salvos em : ' + str(output)
                    else:
                        message = 'Algoritmo retornou seguinte erro:' + str(output)
                        
                    window['-OUTPUT-'].update(message)
                else:
                    window['-OUTPUT-'].update("Selecione arquivos de dados e de configuração antes de executar! ")
                    
        except Exception as e:
            #Display error info to user
            window['-OUTPUT-'].update((str(e)))
                
    window.close()