import PySimpleGUI as sg

class App_GUI:
    def __init__(self) -> None:
        pass
    
    def load(self, cameras_indices):
        dr = ""
        ind = cameras_indices[0]
        layout = [
            [sg.T("")], [sg.Text("Choose an image file: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FileBrowse(key="File")],
            [sg.Text("Camera index: "), sg.OptionMenu(["Camera " + str(w) for w in cameras_indices], default_value=cameras_indices[0], key='Camera')],
            [sg.Button("Start")]
            ]

        ###Building Window
        window = sg.Window('Choose image to edit', layout, size=(600,150))
            
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                break
            elif event == "Start":
                dr = values["File"]
                ind = values['Camera']
                break
        window.close()
        return dr, ind