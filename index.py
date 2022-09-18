from sre_parse import fix_flags
from turtle import onclick
import streamlit as st
import numpy as np

import pickle
from sklearn.preprocessing import StandardScaler
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import time
from datetime import date
import datetime

def home():
    st.write("Welcome")

def detectPregnancyRisk(Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate):

    try:
        Age = float(Age)
        SystolicBP = float(SystolicBP)
        DiastolicBP = float(DiastolicBP)
        BS = float(BS)
        BodyTemp = float(BodyTemp)
        HeartRate = float(HeartRate)
    except:
        return "-1"

    # loading the model
    with open("trained_model/maternalrf.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    # loaded_model = keras.models.load_model("./trained_model/heart12ANN.h5")
    data_array = [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]
    data_array = np.array(data_array,dtype='float64')
    data_array = data_array.reshape(1,-1)
    # sc = StandardScaler()
    # data_array = sc.fit_transform(data_array)
    

    result = loaded_model.predict_proba(data_array)
    max = np.argmax(result[0])
    return max,result[0]

       

    
def PregnancyRiskDetectionModule():
    st.markdown(
        "<h4 style='color:white'> Pregnancy Risk Level Classification</h4>", unsafe_allow_html=True)

    st.write("Please enter values for the following parametrs")
    Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate = 0,0,0,0,0,0
    with st.form(key='my_form'):
        col1, col2 = st.columns([4, 4])
        with col1:
            value = "0"
            Age = st.text_input("Age", value)
            SystolicBP = st.text_input("Systolic Blood Pressure (mmHg)", value)
            DiastolicBP = st.text_input("Diastolic Blood Pressure (mmHg)", value)
            
        with col2:
            BS = st.text_input("Blood Sugar level (mmol/L)", value)
            BodyTemp = st.text_input("Body Temperature (oC)", value)
            HeartRate = st.text_input("Heart Rate (BPM)", value)
           
        if st.form_submit_button("Detect"):
            with st.spinner('Pleas wait...'):
                detected_class, proba  = detectPregnancyRisk(Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate)
                time.sleep(1)
            
            if detected_class != "-1":
            	
                max_value = np.argmax(detected_class)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.write("RESULT")
                if detected_class == 0:
                    st.markdown("<div style='background-color:red'><b>"+str(round(proba[0]*100,2))+"% probability of HIGH pregnancy complication risk level **</b></div><br>"+str(round(proba[1]*100,2)) + "% probability of MEDIUM pregnancy complication risk level<br>"+str(round(proba[2]*100,2)) + "% probability of LOW pregnancy complication risk level", unsafe_allow_html=True)
                elif detected_class == 1:
                    st.markdown("<div style='background-color:yellow;color:black'><b>"+str(round(proba[1]*100,2))+"% probability of MEDIUM pregnancy complication risk level **</b></div><br>"+str(round(proba[0]*100,2)) + "% probability of HIGH pregnancy complication risk level<br>"+str(round(proba[2]*100,2)) + "% probability of LOW pregnancy complication risk level", unsafe_allow_html=True)
                elif detected_class == 2:
                    st.markdown("<div style='background-color:green'><b>"+str(round(proba[2]*100,2))+"% probability of LOW pregnancy complication risk level **</b></div><br>"+str(round(proba[0]*100,2)) + "% probability of HIGH pregnancy complication risk level<br>"+str(round(proba[1]*100,2)) + "% probability of MEDIUM pregnancy complication risk level", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<h6 style='color:red;'>Please enter numeric value</h6>", unsafe_allow_html=True)
        ## saving record **************************************************************************
        st.markdown("<hr>", unsafe_allow_html = True)
        st.markdown("<h4 style='background-color:#333333'>Save Record</h4>", unsafe_allow_html = True)
        nameFile = open("nameList.txt",'r', encoding="utf8")
        nameList = nameFile.readlines()
        nameFile.close()

        option = st.selectbox('Select Patient name', nameList)
        newName = st.text_input('Add New Patient')
        if st.form_submit_button('Save Record'):
            today = date.today()
            d1 = today.strftime("%d/%m/%y")
            if (option!="" and option!="\n") or newName!="":
                if newName!="":
                    with open("nameList.txt", "a") as myfile:
                        myfile.write("\n"+newName)
                    with open(newName+".csv", 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)

                        # write a row to the csv file
                        header = ['visit', 'Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']
                        row = [d1,Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate]
                        writer.writerow(header)
                        writer.writerow(row)
                else:
                    option = option.replace("\n","")
                    with open(option+".csv", 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)

                        # write a row to the csv file
                    
                        row = [d1,Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate]
                        writer.writerow(row)
                st.write("Record saved successfully")
            else:
                st.write("Patient Name not specified! Please add a new Patient or Select from the list")

                    
           

@st.cache
def getHeader():
    image = './images/logo.jpg'
    return image, "Pregnancy monitoring and complication risk level classification System"


def RiksLevelMonitoring():
    st.markdown('<h1>Historical data monitoring', unsafe_allow_html = True)
    nameFile = open("nameList.txt",'r', encoding="utf8")
    nameList = nameFile.readlines()
    nameFile.close()
    with st.form(key = 'monitoring'):
        option = st.selectbox('Select Patient name', nameList)
        if st.form_submit_button("Show"):
            option = option.replace("\n", "")
            data = pd.read_csv(option+".csv")
            # st.bar_chart(data,
            #     columns=data["visit"]
            # )
            col1,col2 = st.columns([6,6]) 
            x_tick = []
            for l in data["visit"]:
                x_tick.append(l)
           
            x = np.array(x_tick)
            default_x_ticks = range(len(x))
            
            SystolicBP_max = [120 for i in range(len(x))]
            SystolicBP_min = [90 for i in range(len(x))]
            
           

            DiastolicBP_max = [80 for i in range(len(x))]
            DiastolicBP_min = [60 for i in range(len(x))]
            
            BS_max = [7.8 for i in range(len(x))]
            BS_min = [3.9 for i in range(len(x))]
            
            BodyTemp_max = [37.5 for i in range(len(x))]
            BodyTemp_min = [35.6 for i in range(len(x))]
            
            HeartRate_max = [100 for i in range(len(x))]
            HeartRate_min = [60 for i in range(len(x))]

            with col1:
                #Systolic Blood Pressure
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.plot(SystolicBP_max, label="Max", color="red")
                ax.plot(data["SystolicBP"], label = "sbp", color="blue")
                ax.plot(SystolicBP_min, label="Min", color="red")
                ax.set_xticks(default_x_ticks, x, fontsize=4,rotation=20, ha="right")
                ax.set_yticks([90,120], fontsize = 5)
                ax.set_title("Systolic Blood Pressure", fontsize = 8)
                ax.set_xlabel("Visit date",  fontsize=4)
                ax.set_ylabel("mmHg", fontsize=4)
                ax.tick_params(axis='y', which='major', labelsize=5)
                ax.tick_params(axis='y', which='minor', labelsize=5)
                ax.fill_between(default_x_ticks, SystolicBP_min, data["SystolicBP"],color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, SystolicBP_max, data["SystolicBP"], color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, SystolicBP_max, data["SystolicBP"], where=(data["SystolicBP"] > SystolicBP_max), color='red', alpha=0.9)
                ax.fill_between(default_x_ticks, SystolicBP_min, data["SystolicBP"], where=(SystolicBP_min > data["SystolicBP"]), color='red', alpha=0.9)
                st.pyplot(fig, figsize=(10,3))

                #Diastolic Blood Pressure
                
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.plot(DiastolicBP_max, label = "Max", color="red")
                ax.plot(data["DiastolicBP"], label = "dbp", color = 'blue')
                ax.plot(DiastolicBP_min, label = "Min", color = 'red')
                ax.set_xticks(default_x_ticks, x,fontsize=4,rotation=20, ha="right")
                ax.set_yticks([60,80])
                ax.set_title("Diastolic Blood Pressure", fontsize=8)
                ax.set_xlabel("Visit date",  fontsize=4)
                ax.set_ylabel("mmHg", fontsize=4)
                ax.tick_params(axis='y', which='major', labelsize=5)
                ax.tick_params(axis='y', which='minor', labelsize=5)
                ax.fill_between(default_x_ticks, DiastolicBP_min, data["DiastolicBP"],color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, DiastolicBP_max, data["DiastolicBP"], color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, DiastolicBP_max, data["DiastolicBP"], where=(data["DiastolicBP"] > DiastolicBP_max), color='red', alpha=0.9)
                ax.fill_between(default_x_ticks, DiastolicBP_min, data["DiastolicBP"], where=(DiastolicBP_min > data["DiastolicBP"]), color='red', alpha=0.9)
                st.pyplot(fig)

                #Blood Sugar level
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.plot(BS_max, label = "Max", color = 'red')
                ax.plot(data["BS"], label = "bsl", color='blue')
                ax.plot(BS_min, label = "Min", color = 'red')
                ax.set_xticks(default_x_ticks, x,fontsize=4,rotation=20, ha="right")
                ax.set_title("Blood Sugar level", fontsize = 8)

                ax.set_xlabel("Visit date",  fontsize=4)
                ax.set_ylabel("mmol/L", fontsize=4)
                ax.tick_params(axis='y', which='major', labelsize=5)
                ax.tick_params(axis='y', which='minor', labelsize=5)
                ax.set_yticks([3.9,7.8])
                ax.fill_between(default_x_ticks, BS_min, data["BS"],color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, BS_max, data["BS"], color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, BS_max, data["BS"], where=(data["BS"] > BS_max), color='red', alpha=0.9)
                ax.fill_between(default_x_ticks, BS_min, data["BS"], where=(BS_min > data["BS"]), color='red', alpha=0.9)
                st.pyplot(fig)
            with col2:

                #Body Temperature
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.plot(BodyTemp_max, label = "Max", color = 'red')
                ax.plot(data["BodyTemp"], label = "bt", color = 'blue')
                ax.plot(BodyTemp_min, label = "Min", color='red')
                ax.set_xticks(default_x_ticks, x,fontsize=4,rotation=20, ha="right")
                ax.set_title("Body Temperature", fontsize = 8)
                ax.set_xlabel("Visit date",  fontsize=4)
                ax.set_ylabel("oC", fontsize=4)
                ax.tick_params(axis='y', which='major', labelsize=5)
                ax.tick_params(axis='y', which='minor', labelsize=5)
                ax.set_yticks([35.6,37.5])
                ax.fill_between(default_x_ticks, BodyTemp_min, data["BodyTemp"],color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, BodyTemp_max, data["BodyTemp"], color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, BodyTemp_max, data["BodyTemp"], where=(data["BodyTemp"] > BodyTemp_max), color='red', alpha=0.9)
                ax.fill_between(default_x_ticks, BodyTemp_min, data["BodyTemp"], where=(BodyTemp_min > data["BodyTemp"]), color='red', alpha=0.9)
                st.pyplot(fig)

                #Heart Rate
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.plot(HeartRate_max, label = "Max", color = 'red')
                ax.plot(data["HeartRate"], label = "hr", color = 'blue')
                ax.plot(HeartRate_min, label = "Min", color = 'red')
                ax.set_xticks(default_x_ticks, x,fontsize=4,rotation=20, ha="right")
                ax.set_yticks([60,100])
                ax.set_title("Heart Rate", fontsize = 8)
                ax.set_xlabel("Visit date",  fontsize=4)
                ax.set_ylabel("bpm", fontsize=4)
                ax.tick_params(axis='y', which='major', labelsize=5)
                ax.tick_params(axis='y', which='minor', labelsize=5)
                ax.fill_between(default_x_ticks, HeartRate_min, data["HeartRate"],color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, HeartRate_max, data["HeartRate"], color='cyan', alpha=0.9)
                ax.fill_between(default_x_ticks, HeartRate_max, data["HeartRate"], where=(data["HeartRate"] > HeartRate_max), color='red', alpha=0.9)
                ax.fill_between(default_x_ticks, HeartRate_min, data["HeartRate"], where=(HeartRate_min > data["HeartRate"]), color='red', alpha=0.9)
                st.pyplot(fig)
def main():
    # ******** header************
    col1, mid, col2 = st.columns([1, 3, 20])
    image, title = getHeader()
    with col1:
        st.image(image, width=100)
    with col2:
        st.markdown("<h3><center>"+title+"</center></h3>",
                    unsafe_allow_html=True)

    # ***********  side bar ****************
    st.sidebar.title("Navigation")
    navigation = st.sidebar.radio("", [
        "Home", "Pregnancy Risk level Predition","Pregnancy risk status monitoring"])

    st.markdown("<hr>", unsafe_allow_html=True)

    if navigation == "Home":
        home()
    elif navigation == "Pregnancy Risk level Predition":
        PregnancyRiskDetectionModule()
    elif navigation == "Pregnancy risk status monitoring":
        RiksLevelMonitoring()

if __name__ == '__main__':
	st.set_page_config(layout = 'wide')
	main()
