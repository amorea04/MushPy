#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:29:13 2021

@author: Adrien
"""
#%% Import packages
import streamlit as st
#import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from st_funct import *
from funct_first_page import *
from funct_second_page import *
from funct_third_page import *
from funct_fourth_page import *
from funct_fifth_page import *
from funct_conclusion_page import *

#%% Load data :
#%%% Set filepaths
global_data = '/Users/Adrien/DataScientist/projet_Mushroom/data_tot_champi.csv'
clean_data = "/Users/Adrien/DataScientist/projet_Mushroom/data_clean_champi.csv"
data_first_model = "/Users/Adrien/DataScientist/projet_Mushroom/data_first_model_w_brightness.csv"
final_data_set = "/Users/Adrien/DataScientist/projet_Mushroom/reduced_dataset_5_families_with_genus.csv"

folderpath = "/Users/Adrien/Documents/GitHub/mushpy/streamlit_mushpy/"

test_imagepath = "/Users/Adrien/Documents/GitHub/mushpy/streamlit_mushpy/images_test"


model_fam_path = "/Users/Adrien/Google Drive/Colab Notebooks/test_save/model_effnetB1_final_20210624.h5"
model_label0_path = "/Users/Adrien/Google Drive/Colab Notebooks/modeles_genus/model_label0_effnetB1_fin_20210627.h5"
model_label1_path = "/Users/Adrien/Google Drive/Colab Notebooks/modeles_genus/model_label1_effnetB1_fin_20210627.h5"
model_label2_path = "/Users/Adrien/Google Drive/Colab Notebooks/modeles_genus/model_label2_effnetB1_fin_20210627.h5"
model_label3_path = "/Users/Adrien/Google Drive/Colab Notebooks/modeles_genus/model_label3_effnetB1_fin_20210627.h5"
model_label4_path = "/Users/Adrien/Google Drive/Colab Notebooks/modeles_genus/model_label4_effnetB1_fin_20210627.h5"


#%%% import dataframes
df, df_clean, df_first, df1 = load_datas(global_data, clean_data, data_first_model, final_data_set)
#%% Pages:
#%% Define multiple pages!    
pages = ["Présentation", "Sélection des données", "Premier cycle de modélisation", 
         "Modèles finaux", "Application", "Conclusion"]
parties = st.sidebar.radio("Navigation :", pages)

#%% partie I

if parties == pages[0]: 
    firstpage(df, df_clean, df_first, df1, folderpath)
    
#%% Partie II

elif parties == pages[1]:
    secondpage_part1(df, df_clean, df_first, df1, folderpath)
    secondpage_part2(df, df_clean, df_first, df1, folderpath)

#%% Partie III

elif parties == pages[2]:
    s1 = f"""
    <style>
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{background-color: #FE9A1A;}}
    </style>
    """
    
    st.markdown(s1, unsafe_allow_html=True)   
    
    options_first_models = ["Classification", "LeNet", "Transfer learning"]
    choix_first = st.sidebar.radio("Choisissez un type de modèle :", options_first_models)
    if choix_first == options_first_models[0]:
        thirdpage_part1(df, df_clean, df_first, df1, folderpath)
    elif choix_first == options_first_models[1]:
        lenet_page(folderpath)
    elif choix_first == options_first_models[2]:
        thirdpage_part2(df, df_clean, df_first, df1, folderpath)

#%% Partie IV

elif parties == pages[3]:
    fourthpage_part1(df, df_clean, df_first, df1, folderpath)

#%% Partie V

elif parties == pages[4]:
    fifthpage(df, df_clean, df_first, df1, folderpath, test_imagepath, model_fam_path,
               model_label0_path, model_label1_path, model_label2_path, model_label3_path,
               model_label4_path)
    

#%% Partie VI

elif parties == pages[5]:
    conclusionpage_part1(folderpath)









