#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:05:42 2021

@author: Adrien
"""
#%% Imports 
import streamlit as st
#import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from st_funct import *
import os
import tensorflow as tf
from tensorflow import keras

#%% fuction for the fourth page

def fifthpage(df, df_clean, df_first, df1, folderpath, test_imagepath, model_fam_path,
               model_label0_path, model_label1_path, model_label2_path, model_label3_path,
               model_label4_path):

    s4 = f"""
    <style>
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{background-color: #973FD0 ;}}
    </style>
    """
    st.markdown(s4, unsafe_allow_html=True)
    
    
#%%% Title section    

    st.markdown("""<div style="color:#973FD0 ; font-size: 34px ;font-weight: bold;">
        Classification de champignons
        </div>
        """, unsafe_allow_html=True)    
    st.write("Les modèles que nous avons mis au point nous permettent de \
             proposer une classification à partir de photos.")

#%%% Load the models
    st.write("Loading Models...")
    model_fam, model_label0, model_label1, model_label2, model_label3, model_label4 = loading_models(model_fam_path, model_label0_path, model_label1_path, model_label2_path, model_label3_path, model_label4_path)
    st.write("Models are loaded")
#%%% Provide a selector for the files    
    def file_selector(folder_path=test_imagepath):
            #filenames = os.listdir(folder_path)
            filenames = [_ for _ in os.listdir(folder_path) if _.endswith(r".jpg")]
            selected_filename = st.selectbox("Choisissez une image", filenames)
            return os.path.join(folder_path, selected_filename)

    filename_imagetest = file_selector()
    st.write("Apperçu de l'image choisie :")
    st.image(filename_imagetest)
#%%% Predict if the button is cliqued
    
    s = f"""
    <style>
    div.stButton > button:first-child {{ border: 5px solid ; border-radius:20px 20px 20px 20px; height: 3em; width: 6em; border-color: #4F8BF9; font-size: 30px; color:#FE9A1A;font-weight: bold;}}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)
    
    if st.button("Predict"):
        fig9 = prediction(filename_imagetest, model_fam, model_label0, model_label1, model_label2, model_label3, model_label4)
        st.pyplot(fig9)
    
    
    
    
    
    
    
    