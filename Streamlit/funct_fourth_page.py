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

#%% fuction for the fourth page

def fourthpage_part1(df, df_clean, df_first, df1, folderpath):
#%%% Définition d'une "couleur de page"    
    s4 = f"""
    <style>
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{background-color: #64DB2D;}}
    </style>
    """
    st.markdown(s4, unsafe_allow_html=True)

#%% Entête de la page    
    st.markdown("""<div style="color:#64DB2D ; font-size: 34px ;font-weight: bold;">
        Modèles finaux :
        </div>
        """, unsafe_allow_html=True)
    
    #st.title("Optimisation du modèle EfficientNetB1 :")

    st.header("Optimisation du modèle EfficientNetB1 :")
    st.markdown(""" <div style='text-align: justify'>
        Compte tenu des performances du modèle que nous avons construit avec EfficentNetB1 
        pour modèle de base (accuracy à 71%), nous avons choisi de poursuivre avec ce modèle.<br>
            <br>
        Ainsi, nous avons choisi de réaliser notre phase d’optimisation de la modélisation en
        travaillant sur 3 aspects :
            <ol>
                <li style="margin-left: 20mm;">L’entraînement et le surentraînement : en variant l’ImageDataGenerator et les couches de dropout.</li>
                <li style="margin-left: 20mm;">Les performances du test : en permettant l’ajustement de la dernière couche de convolution du modèle de base (EfficientNetB1).</li>
                <li style="margin-left: 20mm;">L’interprétabilité en implémentant un algorithme Grad-Cam.</li>
            </ol>
            </div><br>
        """, unsafe_allow_html=True)
 
#%%% Présentation du process d'image data generator         
    st.subheader("1. Image Data Generator :")
    
    imagedata(folderpath)
    
    
#%%% un-freeze des couches de convolution
    st.subheader("2. Entrainement de la dernière couche de convolution :")
    
    st.markdown(""" <div style='text-align: justify'>
        Le principe du <i>transfer learning</i> est de bénéficier d'un entrainement
        long et poussé d'un modèle sur une base de donées très complète.<br>
        Cependant, bien que cet entrainement soit très intéressant, il peut être très 
        bénéfique de permettre l'entrainement des dernières couches de convolution
        afin d'aobtenir un modèle aussi sensible que possible à notre problème 
        spécifique.<br>
        Il faut alors trouver l'équilibre entre les ressources informatiques nécessaires
        et le gain obtenu en permettant l'entrainement de plus en plus de couches de
        convolution.<br><br>
        Dans notre cas nous avons trouvé que <b>l'entrainement de la denière couche de 
        convolution</b> était un bon compromis.
               """, unsafe_allow_html=True)
    
    
    st.markdown("""<div>
                <b><u>Résumé des modèles :</b></u>
                    </div>""", unsafe_allow_html=True)
            
    col1, col2 = st.beta_columns(2)
    col1.write("Modèle initial :")
    imagename2 = folderpath + "20210707_modele_summary_effnetB1_initial.jpg"
    col1.image(imagename2)
    
    col2.write("Modèle final :")
    imagename3 = folderpath + "20210707_modele_summary_effnetB1_final.jpg"
    col2.image(imagename3)
    
    st.markdown(""" <div style='text-align: justify'>
        En établissant une comparaison avec l’architecture utilisée précédemment 
        que nous avons maintenant un nombre bien plus important de paramètres que 
        <b>l’on peut entraîner (trainable params)</b>, une diminution concomitante du nombre de paramètres 
        qu’il n’est pas possible d’entraîner, le tout, comme attendu, avec un nombre 
        de paramètres total qui ne change pas.<br><br>
               </div>""", unsafe_allow_html=True)
    
    st.markdown("""<div>
                <b><u>Résultats :</b></u>
                    </div>""", unsafe_allow_html=True)
    
    col3, col4 = st.beta_columns(2)
    col3.write("Rapport de classification :")
    imagename4 = folderpath + "20210707_Classif_report_modele_efficientnetB1_modele_final.jpg"
    col3.image(imagename4)
    
    col4.write("Matrice de confusion :")
    imagename5 = folderpath + "20210707_confusion_matrix_efficientnetB1_modele_final.png"
    col4.image(imagename5)
    
    st.markdown(""" <div style='text-align: justify'>
        À l’issue de cet entraînement, nous avons donc constaté une amélioration 
        de 6% de l’accuracy sur notre jeu de test (accuracy finale de
        <a style="color:#64DB2D ; font-weight: bold;">77%</a> 
        !), ce qui est une très nette amélioration par rapport au modèle initial.<br><br>
               </div>""", unsafe_allow_html=True)    
    
    st.text("Caractéristiques du modèle en fin d'entrainement : \n\n\
            loss: 0.6498 - accuracy: 0.7423 - val_loss: 0.6500 - val_accuracy: 0.7754")
    
#%%% Interprétabilité avec le grad-cam      

    st.subheader("3. Interprétabilité, Grad-CAM :")
    
    st.markdown(""" <div style='text-align: justify'>
        Finalement, afin de comprendre sur quelle base notre modèle s’appuyait pour 
        réaliser les classifications nous nous sommes tournés vers l’algorithme Grad-CAM.
        Ce dernier est l’acronyme de Gradient-weighted Class Activation Map développé et 
        publié par Ramprasaath R. Selvaraju en 2017 (<i>Grad-CAM: Visual Explanations 
       from Deep Networks via Gradient-based Localization, 2017</i>). Cette approche 
        provient d’une catégorie plus générale qui consiste à produire des heatmaps 
        représentant les classes d’activation sur les images d’entrée. Une classe 
        activation heatmap est associée à une classe de sortie spécifique. Ces classes 
        sont calculées pour chaque pixel d’une image d’entrée, indiquant l’importance 
        de chaque pixel par rapport à la classe considérée.<br>
        <br>
        En d’autres termes, il va être possible d’attribuer à chaque pixel son 
        importance dans le processus de décision permettant d’attribuer la classe à 
        l’objet.<br><br>
               </div>""", unsafe_allow_html=True)        
        
    imagename = folderpath + "process_GradCam.jpeg"
    st.image(imagename)
    st.caption("Images d’illustration pour la compréhension du fonctionnement de Grad-CAM")    

    st.markdown("____")
    
#%%% Modélisation entonoir !

    st.header("Prédiction du genre :")
    
    st.markdown(""" <div style='text-align: justify'>
        Une piste très intéressante d’amélioration réside dans l’objectif d’atteindre 
        une classification plus fine. Initialement, nous avons choisi de nous focaliser 
        sur l’échelle des familles dans la classification, notamment pour des raisons 
        d’équilibre entre le nombre d’images et le nombre de classes que nous 
        souhaitions prédire.<br><br>
               </div>""", unsafe_allow_html=True)

    st.subheader("1. Présentation des données :")

    st.markdown(""" <div style='text-align: justify'>
        Afin de générer de nouveaux jeux de données pour entraîner 5 nouveaux modèles, 
        nous nous sommes basés sur le jeu de données initial comprenant les 5 familles. 
        Pour chaque famille, nous nous sommes alors intéressés au genre et avons exploré 
        un peu les données. Puisque certains genres possédaient que très peu d’images, 
        nous avons choisi de ne conserver que les genres possédant plus de 100 
        images.<br><br>
               </div>""", unsafe_allow_html=True)

    imagename6 = folderpath + "20210709_genus_repartition.png"
    st.image(imagename6)
    st.caption("Répartition des genres pour chaque famille.")
    
    st.subheader("2. Entrainement des modèles :")
    
    st.markdown("""<div>
                <b><u>Rapport de classification pour les modèles :</b></u>
                    </div>""", unsafe_allow_html=True)
    

    imagename7 = folderpath + "20210709_Classif_report_modele_efficientnetB1_fam0_genus.jpg"
    imagename8 = folderpath + "20210709_Classif_report_modele_efficientnetB1_fam1_genus.jpg"
    imagename9 = folderpath + "20210709_Classif_report_modele_efficientnetB1_fam2_genus.jpg"
    imagename10 = folderpath + "20210709_Classif_report_modele_efficientnetB1_fam3_genus.jpg"
    imagename11 = folderpath + "20210709_Classif_report_modele_efficientnetB1_fam4_genus.jpg"

    col5, col6 = st.beta_columns(2)
    col5.image(imagename7)
    col6.image(imagename8)
    st.markdown(""" <br> """, unsafe_allow_html=True)
    col7, col8 = st.beta_columns(2)
    col7.image(imagename9)
    col8.image(imagename10)
    st.markdown(""" <br> """, unsafe_allow_html=True)
    col9, col10 = st.beta_columns(2)
    col9.image(imagename11)

    st.markdown(""" <div style='text-align: justify'>
        <br>Nous pouvons constater que nos modèles ne souffrent pas particulièrement de 
        surapprentissage et en plus que nous bénéficions d’une accuracy plutôt bonne 
        (toujours supérieure à 
        <a style="color:#64DB2D ; font-weight: bold;">82 %</a>).
         Il faut cependant se remémorer que nous avons des 
        classes plutôt déséquilibrées. Nous avons veillé à utiliser l’ImageDataGenerator 
        car il est indiqué dans la littérature que cette méthode permet notamment de 
        contrer ce déséquilibre. Nous pouvons aussi remarquer que l’efficacité des modèles 
        reste meilleure que le hasard (même avec ce déséquilibre)<br><br>
               </div>""", unsafe_allow_html=True)










