#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 21:44:38 2021

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
from matplotlib import legend
from st_funct import *

#%% fuction for the second page
def secondpage_part1(df, df_clean, df_first, df1, folderpath):
    
    s2 = f"""
    <style>
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{background-color: #3DAFD6;}}
    </style>
    """
    st.markdown(s2, unsafe_allow_html=True)
    
#%%% En tête :
    
    st.markdown("""<div style="color:#3DAFD6;font-size: 34px;font-weight: bold;">
        Sélection des données
        </div>
        """, unsafe_allow_html=True)
    
    st.title("")
    
    st.write("En choisissant les données labellisées \"clean\" nous avions \
             encore", df_clean.shape[0], "images.")
        
    st.markdown(""" <div style='text-align: justify'>
        Compte tenu de l'équilibre des données, nous avons hésité à nous focaliser sur :
            <ul>
                <li style="margin-left: 20mm;">le <b>genre</b> <i>ou</i></li>
                <li style="margin-left: 20mm;">la <b>famille</b></li>
            </ul>
            Nous avons donc choisi de nous focaliser sur la <b>famille</b>, avec l'idée
            de proposer à terme une modélisation en cascade avec dans un premier temps
            une prédiction sur la famille, puis une prédiction du genre.
            </div><br>
        """, unsafe_allow_html=True)
    
    st.markdown("____")
#%%% Tri des données    
    st.header("Tri des données :")

    st.markdown(""" <div style='text-align: justify'>
        Dans un premier temps, nous avons donc réalisé un premier tri 
        basé exclusivement sur la disponibilité des images : seules les 
        familles disposant de plus de 3000 images chaccune ont été retenues.
            </div><br>
        """, unsafe_allow_html=True)
    
    st.write("Ce qui nous a permi d'obtenir", df_first.shape[0], "images réparties :")

    #st.dataframe(df_first.head(10))
    fig5 = plt.figure(figsize = (12, 7))
    sns.countplot(df_first['family']);
    plt.xticks(rotation = 90);
    st.pyplot(fig5)

             
    st.markdown(""" <div style='text-align: justify'>
        Après avoir lancé la permière modélisation, nous nous sommes
             confrontés aux limtes de puissances de calcul nécessaires à ce
             type de projet.
            </div><br>
        """, unsafe_allow_html=True)


    st.markdown("____")
    
#%%% Jeu de données retenu :    
    st.header("Jeu de données finalement retenu :")
    
    st.write("Nous avons donc été contraint de réduire le jeu de données pour \
            finalement obtenir", df1.shape[0], "images réparties :")
    
    fig6 = plt.figure(figsize = (12, 7))
    sns.countplot(df1['family'], palette="crest");
    plt.xticks(rotation = 70);
    st.pyplot(fig6)
    
    st.markdown("____")
    
#%%% Etude de la luminosité des images
    st.header("Etude de la luminosité des images :")
    
    st.markdown(""" <div style='text-align: justify'>
        La luminosité des images est un paramètre très facilement capté
             par les modèles de machine/deep learning. Il est donc crucial de
             s'assurer que notre jeu de données ne contient pas ce biais.
            </div><br>
        """, unsafe_allow_html=True)
    
    liste1 = ['Inocybaceae','Omphalotaceae','Fomitopsidaceae','Physalacriaceae','Marasmiaceae']
    df2 = df_first[df_first['family'].isin(liste1)]
    df2 = df2.reset_index(drop=True)
    #st.dataframe(df2)
    
    fig7 = plt.figure(figsize = (12, 7))
    fig7 = sns.displot(data=df2, x='brightness', hue='family', kind='kde', fill=True, height=8, aspect=1.8)
    st.pyplot(fig7)
             
    st.markdown(""" <div style='text-align: justify'>
        Nous pouvons donc constater que toutes les familles possèdent 
             des images dont la luminosité similaire. Notre jeu de données 
             ne présente donc pas ce biais (ce ne sera pas un souci pour les 
             étapes de modélisation).
            </div><br>
        """, unsafe_allow_html=True)             

    st.markdown("____")



#%%
def secondpage_part2(df, df_clean, df_first, df1, folderpath):
    # %%% En tête :
    st.header("Visualisation des données")

    st.subheader("Visualisation des familles à l’oeil humain")

    st.markdown("""<div style='text-align: justify'>
            Désormais, il est intéressant de se familiariser avec notre jeu de données
            et de connaître les images qu'il contient. Le but est de voir si nous pouvons 
            à l'œil voir des différences entre nos classes et savoir si nous-mêmes pouvons 
            classer facilement ces familles. <br><br>
            Ci-dessous, vous trouverez différentes photos de nos classes prises
            aléatoirement afin de nous “entraîner” sur nos classes.<br><br></div>
        """, unsafe_allow_html=True)
    st.write("\n")

    imagepath1 = folderpath + "comparison_labels.png"
    st.image(imagepath1, width=750)

    st.markdown("""<div style='text-align: justify'><br>
    Nous pouvons remarquer plusieurs choses: <br></div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <ul>
            <li>La classification est difficile pour un œil non entraîné</li>
            <li>Le label 2 est lui bien distinct des autres et facilement reconnaissable
                <ul>
                    <li>Le label 2 doit avoir de bonnes performances sur nos modèles</li>
                </ul>
            </li>
        </ul>
        """, unsafe_allow_html=True)


    st.subheader("Réduction de dimensions")

    st.markdown("""<div style='text-align: justify'>
    Il est maintenant intéressant d’appliquer \
    la méthode de réduction de dimensions à nos images.<br> \
    Cela peut déjà faire apparaître des clusters de famille et nous aider à sélectionner \
    des features si tel est le cas.<br><br>
    Après avoir réaliser un prétraitement des images, nous avons appliqué les méthodes de réductions \
    de dimensions Isomap et PCA:</div>
        """, unsafe_allow_html=True)
    st.write("\n")

    options_eq3 = ["Isomap", "PCA"]
    choix_eq3 = st.radio("Choisissez un clade :", options_eq3)

    image_eq3 = plot_eq3(choix_eq3, folderpath)
    st.image(image_eq3, width=550)

    st.markdown("""<div style='text-align: justify'>L’utilisation de ces deux méthodes a démontré \
     que nous ne pouvions pas mettre en évidence de clusters de famille avec une méthode de réduction. \
     Néanmoins, ce résultat est à modérer, car nous avons énormément réduit les dimensions jusqu’à en \
     garder que deux.</div>
        """, unsafe_allow_html=True)




             
             
             