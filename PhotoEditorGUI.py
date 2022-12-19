import PySimpleGUI as sg

class App_GUI:
    def __init__(self) -> None:
        pass
    
    def load_file(self):
        
        dr = ""
        layout = [[sg.T("")], [sg.Text("Choose a folder: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]

        ###Building Window
        window = sg.Window('My File Browser', layout, size=(600,150))
            
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                break
            elif event == "Submit":
                dr = values["-IN-"]
                break
        window.close()
        return dr