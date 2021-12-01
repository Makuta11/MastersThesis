#Copy to terminal 
import cv2
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import html
from dash import dcc

from threading import Thread
from dash.dependencies import Output, Input, State

import time
import serial
import pickle
import requests
import webbrowser

from utils.fetchLinks import *

hostName = "192.168.0.100" #Egmont
#hostName = "10.11.131.110" #DTU

class linkHolder():
    def __init__(self):
        self.RestDir = ""
        self.TaskDir = ""
        self.Task2Dir = ""

"""Webcam Recording Class""" 
class recordThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.participant = ""
        self.session = ""

    def run(self):
        self.running = True                                     # bool to control while loop
    
    def record(self,task,n_clicks):
        self.counter = 0
        self.running = True
        self.cap = cv2.VideoCapture(0)                          # start camera stream

        self.video = []
        self.times = np.zeros(27000)                            # initialize times vector (max 15min)

        while self.running:
            # Capture frame-by-frame
            self.times[self.counter] = time.time()              # capture time stamp
            ret, frame = self.cap.read()                        # capture frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # convert to grayscale
            if ret:
                self.video.append(frame)                        # append frame to video
            self.counter += 1                                   # add one to counter
        
        print("saving video")
        self.times = self.times[self.times!=0]                  # dismiss all zero timestamps
        
        t = time.time()
        with open(f'{OutputDir}/{self.participant}_ses:{self.session}_{task}_video_{n_clicks}.npy', 'wb') as fp:        # save the video 
            pickle.dump(self.video, fp)        
        print(f'Video Save Time: {round(time.time() - t, 3)}sec')
        
        t = time.time()
        with open(f'{OutputDir}/{self.participant}_ses:{self.session}_{task}_time_stamps_{n_clicks}.npy', 'wb') as fp:  # save the time stamps
            pickle.dump(self.times, fp)                     
        print(f'Time Stamps Save Time: {round(time.time() - t, 3)}sec')

        self.cap.release()                                      # stop camera stream
        self.video_nr += 1                                      # add one to the video itterator                                                        
        print("Thread stopped")

    def terminate(self):
        self.running = False
        print(self.running)

class lampThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        self.running = True
    
    def lampTurnOn(self):
        self.running = True
        self.time0 = time.time()
        requests.get(f"https://{hostName}:5932/turn-on", verify=False)
        while self.running and (time.time() - self.time0 < 15*60):
            pass
        
        print("While loop over")
        requests.get(f"https://{hostName}:5932/turn-off", verify=False)

    def lampTurnOff(self):
        self.running = False

#### Import of Montserrat Font ####
external_stylesheets = [{
        "href": "https://fonts.googleapis.com/css2?"
        "family=Montserrat:wght@400;700&display=swap",
        "rel": "stylesheet"}
]

#### Dashboard Layout ####
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

button__default = {'transition-duration': '0.4s',
                    'font-size': '40px',
                    'border-radius': '12px',
                    'padding': '15px',
                    'color': 'white',
                    'background-color': '#e2bf6f',
                    'border': '0'}
button__clicked = {'transition-duration': '0.2s',
                    'font-size': '40px',
                    'border-radius': '12px',
                    'padding': '15px',
                    'color': '#e2bf6f',
                    'background-color': '#414b65',
                    'border': '0'}

app.title = "Optoceutics Data Collection"

#### Page Renderer ####
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ]
)

#### Main Page ####
page_main = html.Div([
        #### header start ####
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src = app.get_asset_url('OClogo.png'),
                            className='app__header__img'
                        ),
                    ],
                    className = "app__header__logo"
                ),
                html.Div(
                    [
                        html.H1(
                            children="Data Collection Dashboard", className="app__header__title"
                        ),
                    ], 
                    className = "app__header_DASHBOARD"
                )
            ], 
            className = "app__header",
        ), 

        #### container start ####
        html.Div([
            # Rec1
            html.Div(
                [
                    # text
                    html.Div(
                        children="Resting State EEG 1", className="containerColumn"
                    ),
                    # button
                    html.Div(
                        [
                            html.Button('Start', id="EEG-button", n_clicks=0,className="button"),
                            dcc.Store(id="EEG-button-store",storage_type="local")
                        ], className="button_container"
                    )
                ], className="containerRow"
            ),
            # Rec2
            html.Div(
                [
                    # text
                    html.Div(
                        children="N-back Test 1", className="containerColumn"
                    ),
                    # button
                    html.Div(
                        [
                            html.Button('Start', id="TASK-button", n_clicks=0,className="button")
                        ], className="button_container"
                    )
                ], className="containerRow"
            ),
            # Rec3
            html.Div(
                [
                    # text
                    html.Div(
                        children="Stimulation Period", className="containerColumn"
                    ),
                    # button
                    html.Div(
                        [
                            html.Button('Start', id="STIM-button", n_clicks=0,className="button")
                        ], className="button_container"
                    ),dcc.Store(id="button-bool-1"),dcc.Store(id="button-bool-2")
                ], className="containerRow"
            ),
            # Rec4
            html.Div(
                [
                    # text
                    html.Div(
                        children="Resting State EEG 2", className="containerColumn"
                    ),
                    # button
                    html.Div(
                        [
                            html.Button('Start', id="EEG-button-2", n_clicks=0,className="button"),
                            dcc.Store(id="EEG-button-store-2",storage_type="local")
                        ], className="button_container"
                    )
                ], className="containerRow"
            ),
            # Rec5
            html.Div(
                [
                    # text
                    html.Div(
                        children="N-back Test 2", className="containerColumn"
                    ),
                    # button
                    html.Div(
                        [
                            html.Button('Start', id="TASK2-button", n_clicks=0,className="button")
                        ], className="button_container"
                    ),dcc.Store(id="button-bool-3")
                ], className="containerRow"
            ),
        ], className="app__container"
        ),
        html.Div([
            html.A(html.Button("Session Settings",className="SessionSettingsButton",style={"position": "absolute","bottom": "0","margin": "25px"}), href="/page-settings",className="SessionSettings")
        ]),
        html.Div([
                html.Button("Fetch Links",id="links",n_clicks=0,className="SessionSettingsButton",style={"position": "absolute","bottom": "60px","margin": "25px"})
        ]), dcc.Store("link-bool")
    ]
)

# Fetch Links
@app.callback(
    Output("link-bool","value"),
    [Input("links", 'n_clicks')]
)
def fetching(n_clicks):
    if n_clicks > 0:
        print("fetching")
        links.RestDir, links.TaskDir, links.Task2Dir = fetchLinks()
        print("Done")
    return True

# EEG button
@app.callback([
    Output("EEG-button","children"),
    Output("EEG-button","style")],
    [Input("EEG-button", 'n_clicks')]
)
def rec_button(n_clicks):
    click = n_clicks%2
    if click == 0:
        if serCon and n_clicks > 0: #Send trigger to ZETO
            ser.write(b'H')
        if n_clicks > 0:
            RS1_timestamps.append(time.time())
            with open(f'{OutputDir}/{recording.participant}_ses:{recording.session}_RS1_time_stamps_{(n_clicks/2) - 1}.npy', 'wb') as fp:        # save the video 
                pickle.dump(RS1_timestamps, fp)   
        return f"Start", None
    elif click == 1:
        if serCon: #Send trigger to ZETO
            ser.write(b'H')
        RS1_timestamps.append(time.time())
        webbrowser.open(links.RestDir)
        return f"Stop", button__clicked

# EEG button 2
@app.callback([
    Output("EEG-button-2","children"),
    Output("EEG-button-2","style")],
    [Input("EEG-button-2", 'n_clicks')]
)
def rec_button(n_clicks):
    click = n_clicks%2
    if click == 0:
        if serCon and n_clicks > 0: #Send trigger to ZETO
            ser.write(b'H')
        if n_clicks > 0:
            RS2_timestamps.append(time.time())
            with open(f'{OutputDir}/{recording.participant}_ses:{recording.session}_RS2_time_stamps{(n_clicks/2) - 1}.npy', 'wb') as fp:        # save the video 
                pickle.dump(RS2_timestamps, fp)
        return f"Start", None
    elif click == 1:
        if serCon: #Send trigger to ZETO
            ser.write(b'H')
        RS2_timestamps.append(time.time())
        webbrowser.open(links.RestDir)
        return f"Stop", button__clicked

# TASK 1 Button
@app.callback([
    Output("TASK-button","children"),
    Output("TASK-button","style")
    ],[Input("TASK-button", 'n_clicks')]
)
def rec_button(n_clicks):
    click = n_clicks%2
    if click == 0:
        if serCon and n_clicks > 0: #Send trigger to ZETO
            ser.write(b'H')
        return f"Start", None
    elif click == 1:
        if serCon: #Send trigger to ZETO
            ser.write(b'H')
        return f"Stop", button__clicked

# Recorord from TASK 1 button
@app.callback(
    Output("button-bool-1","data"),
    [Input("TASK-button", 'n_clicks')]
    )
def recordVideo(n_clicks):
    if n_clicks%2 == 1:
        print("video start")
        webbrowser.open(links.TaskDir)
        recording.record("N-Back-1",(n_clicks-1)/2)
        return n_clicks

    elif n_clicks > 1 and n_clicks%2 == 0:
        print("terminating")
        recording.terminate()
        return n_clicks

# TASK 2 Button
@app.callback([
    Output("TASK2-button","children"),
    Output("TASK2-button","style")
    ],[Input("TASK2-button", 'n_clicks')]
)
def rec_button(n_clicks):
    click = n_clicks%2
    if click == 0:
        if serCon and n_clicks > 0: #Send trigger to ZETO
            ser.write(b'H')
        return f"Start", None
    elif click == 1:
        if serCon: #Send trigger to ZETO
            ser.write(b'H')
        return f"Stop", button__clicked

# Recorord from TASK 2 button
@app.callback(
    Output("button-bool-3","data"),
    [Input("TASK2-button", 'n_clicks')]
    )
def recordVideo(n_clicks):
    if n_clicks%2 == 1:
        print("video start")
        webbrowser.open(links.Task2Dir)
        recording.record("N-Back-2",(n_clicks-1)/2)
        return n_clicks

    elif n_clicks > 1 and n_clicks%2 == 0:
        print("terminating")
        recording.terminate()
        return n_clicks

# light control from STIM button
@app.callback(
    Output("button-bool-2","data"),
    [Input("STIM-button", 'n_clicks')]
    )
def recordVideo(n_clicks):
    if n_clicks%2 == 0:
        print("Lamp Off")
        lamp.lampTurnOff()
        return n_clicks

    elif n_clicks%2 == 1:
        print("terminating")
        lamp.lampTurnOn()
        return n_clicks

# STIM button
@app.callback([
    Output("STIM-button","children"),
    Output("STIM-button","style")],
    [Input("STIM-button", 'n_clicks')]
)
def STIM_button(n_clicks):
    click = n_clicks%2
    if click == 0:
        if serCon and n_clicks > 0: #Send trigger to ZETO
            ser.write(b'H')
        lamp.lampTurnOff()
        return f"Start" , None
    elif click == 1:
        if serCon: #Send trigger to ZETO
            ser.write(b'H')
        STIM_timestamps.append(time.time())
        return f"Stop" , button__clicked

page_settings = html.Div([
html.Div(
    [
        html.Div([
        html.Div(
            [
            # text
                html.Div(
                    children="Participant:", className="containerInput"
                ),
                # Input
                html.Div(
                    [
                        dcc.Input(id="input_participant", type="text", persistence= True, persistence_type="session", placeholder="Participant name", className="inputStyle",style={'width': '90%'})
                    ], className="input_container"
                ),dcc.Store(id="participant",storage_type="local")
            ], className="containerRow"
        ),
        html.Div(
            [
            # text
                html.Div(
                    children="Session Nr.:", className="containerInput"
                ),
                # Input
                html.Div(
                    [
                        dcc.Input(id="input_session", type="text", persistence= True, persistence_type="session", placeholder="Session number", className="inputStyle",style={'width': '90%'})
                    ], className="input_container"
                ),dcc.Store(id="sessionNr",storage_type="local")
            ], className="containerRow"
        )
    ], style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -70%)"})
    ], className="app__container"
),
html.Div(
    [
        html.A(html.Button("Go Back",className="button",style={"position": "absolute","bottom": "0","margin": "25px"}), href="/")
    ]
)
])

# Participant Callback
@app.callback(
    Output("participant","value"),
    [Input("input_participant", 'value')]
)
def saveParticipantName(participantName):
    recording.participant = participantName
    return participantName

# Session Callback
@app.callback(
    Output("sessionNr","value"),
    [Input("input_session", 'value')]
)
def saveParticipantName(sessionNumber):
    recording.session = sessionNumber
    return sessionNumber

# Page Navigation Button
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-settings':
        return page_settings
    else:
        return page_main

if __name__ == "__main__":
    OutputDir = "/Users/DG/Documents/PasswordProtected/Speciale Outputs"
    recording = recordThread()
    recording.start()
    lamp = lampThread()
    lamp.start()
    links = linkHolder()
    RS1_timestamps = list()
    RS2_timestamps = list()
    STIM_timestamps = list()
    try:
        ser = serial.Serial('/dev/cu.usbmodem1101', 9600)#'/dev/tty.usbmodem1201', 9600)
        print("Serial Connected")
        serCon = True
    except:
        print("Serial not available")
        serCon = False
    app.run_server(debug=True)
