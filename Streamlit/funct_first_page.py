#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:02:26 2021

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

#%% fuction for the first page

def firstpage(df, df_clean, df_first, df1, folderpath):
#%%% Title section
        
    st.title('Projet Mushpy')
    
    st.write("Parcours Bootcamp - Promo mai 2021 - Data Scientist")
    
    imagepath1 = folderpath + "111.jpg"
    st.image(imagepath1, width=550)


    st.markdown(""" <div style='text-align: justify'><br>
        <b>D'après :</b>
            <ul>
                <li style="margin-left: 14mm;">Sevil CARON</li>
                <li style="margin-left: 14mm;">Adrien MOREAU</li>
                <li style="margin-left: 14mm;">Fernando GONÇALVES</li>
                <li style="margin-left: 14mm;">Thibault KACZMAREK</li>
            </ul>
            </div><br>
        """, unsafe_allow_html=True)             
    
    st.write("[Projet Github] (https://github.com/DataScientest/mushpy)")
    st.markdown("____")
    
#%%%% Drop few columns for the analysis
    df_temp = df.drop(["name_0", "species_1", "genus_2", "family_3", "order_4", "classe_5", "phylum_6"]
                      ,axis = 1)

#%%% Exploration des données    
    st.header("Données :")
    
    st.write("Dans un premier temps, regardons les données dont nous disposons:")
    
    st.write(df_temp.head())
    
# %%% Création de la sidebar avec les variables que l'on retrouve dans le dataset
       
    st.sidebar.header('Variables du jeu de données')
    st.sidebar.write(df_temp.columns)

#%%% Présentation des données dans le dictionnaire    
    st.subheader("Nous pouvons consater que les données sont contenues dans un dictionnaire :")
    
    dict_exemple = pd.DataFrame.from_dict(ast.literal_eval(df["gbif_info"].iloc[0]), orient='index')
    
    st.write(dict_exemple)

    st.markdown(""" <div style='text-align: justify'>
        A partir de ce dictionnaire, nous avons donc choisi d'extraire le nom, 
        l'espèce, le genre, la famille, l'ordre, la classe et le phylum.
            </div><br>
        """, unsafe_allow_html=True)

    st.markdown("____")              
    
    #%%% Pretraitement antérieur

    st.header("Données \"pré-analysées\" :")
    
    st.subheader("Les données avaient préalablement été explorées :")
    
    st.markdown(""" <div style='text-align: justify'>
        Dans les données initiales, nous avons également pu constater qu'il y
        avait une colonne "thumbnail". L'étiquette "thumbnail" correspond à une
        analyse préalablement réalisée qui avait déterminé des données dites propres.
            </div><br>
        """, unsafe_allow_html=True)        
    
    imagepath2 = folderpath + "thumbnail.png"
    st.image(imagepath2, width=550)
    
    st.write("Nous pouvons constater qu'un tier du jeu de données peut-être considéré comme \"propre\". \n")
    
    st.subheader("Niveau taxonomique du label :")

    st.markdown(""" <div style='text-align: justify'>
            Les donnéees sont issues du site 
            <a href="http://mushroomobserver.org">MushroomObserver</a>.
            Sur ce site, une communauté d'amateurs indentifie les champignons proposés sur 
            les photos. Ainsi, l'identification d'un champignon repose sur le consensus 
            atteint par la communauté. Les données sont donc "labellisées", néanmoins 
            le niveau taxonomique du label n'est pas uniformisé pour tous les champignons :
            </div><br>
           """, unsafe_allow_html=True)
    
    imagepath3 = folderpath + "label_rank.png"
    st.image(imagepath3, width=550)

    st.markdown(""" <div style='text-align: justify'>
            Nous pouvons donc constater que l'espèce et le genre constitient 
             les rang taxonomiques (les plus précis) les plus fréquemment utilisés par les
             amateurs pour identifier les champignons.
            </div><br>
           """, unsafe_allow_html=True)
    
    
    st.subheader("Analyse des valeurs manquantes :")
    
    data_nan2 = nan_analysis(df, df_clean)
    
    st.dataframe(data_nan2)


    st.markdown(""" <div style="text-align: justify; font-weight:bold;">
            <br>Puisque nous pouvons bénéficier d'une "pré-analyse", nous
             avons décidé de l'utiliser pleinement et de nous focaliser
                 sur les données indiquées comme "clean" (thumbnail = 1).
            </div><br>
           """, unsafe_allow_html=True)
    
    st.markdown("____")
    # %%% Combien avons nous d'espèces ou de familles différentes ?
    st.header("Homogénéité et équilibre des données")
    
    exp_eq = st.beta_expander("Répartition des données \"Phylum\", \"Classe\", \"Ordre\" : (max. 30 entrées)")
    
    options_eq = ["Phylum", "Classe", "Ordre"]
    choix_eq = exp_eq.radio("Choisissez un clade :", options_eq)
    
    image_eq = plot_eq(choix_eq, folderpath)
    exp_eq.image(image_eq, width=600)

    exp_eq.markdown(""" <div style='text-align: justify'>
        Nous constatons facilement que les rang Phylum, classe et ordre sont :
            <ol>
                <li style="margin-left: 20mm;">Très mal équilibrées,</li>
                <li style="margin-left: 20mm;">sont trop générale (rang phylogénique trop élevé)</li>
            </ol>
            </div><br>
        """, unsafe_allow_html=True)
    
    st.subheader("Répartition des données \"Famille\", \"Genre\" : (max. 30 entrées)")
    
    options_eq2 = ["Famille", "Genre"]
    choix_eq2 = st.radio("Choisissez un clade :", options_eq2)
    
    image_eq2 = plot_eq2(choix_eq2, folderpath)
    st.image(image_eq2, width=750)

    st.markdown(""" <div style='text-align: justify'>
        <b>A l'opposé, les données au niveau du genre et de la famille sont :</b>
            <ol>
                <li style="margin-left: 20mm;">Moins générales,</li>
                <li style="margin-left: 20mm;">Présentent certaines classes mieux équilibrées</li>
            </ol>
            </div><br>
        """, unsafe_allow_html=True)

        
    st.markdown("____")





