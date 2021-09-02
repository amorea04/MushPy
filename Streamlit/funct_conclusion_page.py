#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:10:00 2021

@author: Fernando
"""
#%% Imports 
import streamlit as st

def conclusionpage_part1(folderpath):

#%%% Changement de couleur des boutons (sidebar)    
    s5 = f"""
    <style>
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{background-color: #BBDE20;}}
    </style>
    """
    st.markdown(s5, unsafe_allow_html=True)
    
#%%% En tête :
    
    st.markdown("""<div style="color:#BBDE20;font-size: 34px;font-weight: bold;">
        Conclusion
        </div>
        """, unsafe_allow_html=True)
    
    #st.title("Conclusion")

    st.header("Déroulement d’un projet :")

    st.markdown("""<div style='text-align: justify'>
        Ce projet nous a permis d’aborder l’intégralité du déroulement d’un projet en Data Science. <br>
        Avec cette problématique de classification nous avons pu :</div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <ul>
            <li>construire notre stratégie</li>
            <li>mettre en pratique nos nouvelles compétences</li>
            <li>exploré des pistes
                <ul>
                    <li>réductions de <b>dimensions</b> (plus classification)</li>
                    <li>architecture <b>LeNet</b></li>
                </ul>
            </li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("""
        <br>
        Nous avons été confrontés à plusieurs problématiques :<br/>
        <ul>
            <li>choix des données</li>
            <li>implémentation d’un algorithmes</li>
            <li>choix et optimisation de modèles</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("""
        <br>
        <i>In-fine</i>, le <b>transfer learning</b> a été retenu parce que plus efficace pour résoudre notre problème de classification.<br>
        Nos prédictions sont correctes dans plus de <b>75%</b> des cas.
        <br><br>
        La marge de progrès reste importante.
        """, unsafe_allow_html=True)

    st.markdown("""
        <br>                
        Il faut garder en mémoire que nous nous sommes concentrés uniquement sur une
        partie (très) réduite de notre jeu de données : <br>
        <br>
        <bq><bq><b>5 familles</b>, ne représentantque <b>17.000 images</b> <br>
        <br>
        sur les <b>650.000</b> images initialement disponibles, Nous avons retenu :
        <br>
        <br>        
        """, unsafe_allow_html=True)

    imagepath = folderpath + "images_count.png"
    st.image(imagepath)

    st.markdown("""
        <div style='text-align: justify'>
        Nous aurions pu essayé d'écrêter lors du chargement (en vue d'utiliser un <i>ImageDataGenerator</i> avec <i>flow_from_directory</i> par exemple) le nombre d'images par familles afin d'avoir :
        <ul>
            <li>un dataset équilibré</li>
            <li>plus d'images</li>
            <li>plus de classes</li>
        </ul>
        
        <div style='text-align: justify'>Les ressources à notre dispositions étaient hélas insuffisantes.<br>
        Il était difficile de travailler avec plus de 4 ou 5 familles (après cela, le nombre d'images et donc les temps de calculs devenaientt trop important').
        Ce point limite la possibilité d'une classification directe plus fine (au niveau du genre par exemple).</div>
        </div>
        <br><br>
        """, unsafe_allow_html=True)

    imagepath = folderpath + "famille_eq_cut.png"
    st.image(imagepath)


    st.markdown("""
        <br>
        <div style='text-align: justify'>
        Il est à noter que certaines images sont de mauvaise qualité ou ne sont pas réellement des photos de champignons.
        Une première piste d’amélioration (et probablement une piste cruciale) réside dans l’amélioration de celles-ci.
        <div>
        """, unsafe_allow_html=True)

    st.markdown("""
        """, unsafe_allow_html=True)

    st.markdown("""
        """, unsafe_allow_html=True)

    

    st.markdown("____")

    st.header("Possibles application :")

    st.markdown("""
        <div style='text-align: justify'>
        En l’état actuel les applications de notre modèle sont relativement limitées tant la classification ne concerne qu’une très faible quantité des familles existantes. Néanmoins, en accédant à des ressources informatiques plus conséquentes, il serait possible de développer plus en profondeur cette approche et ainsi élargir le spectre de la prédiction.
        <div>
        """, unsafe_allow_html=True)

    st.markdown("""
        """, unsafe_allow_html=True)

    st.markdown("____")
    
    
    
    
    
    
    
    
    
