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

#%% fuction for the third page
def thirdpage_part1(df, df_clean, df_first, df1, folderpath):
    st.markdown("""<div style="color:#FE9A1A;font-size: 34px;font-weight: bold; line-height:45px;">
        Essais de modélisation par réduction de dimension :
        </div>
        """, unsafe_allow_html=True)  

    st.markdown("""<div style='text-align: justify'>Bien que la réduction n'est pas été concluante \
            quant à la mise en évidence de clusters sur des plans, \
            il est intéressant d'aller au bout de notre démarche et de \
            tester une modélisation grâce à cette méthode. <br><br>
            La méthode employée a consisté en: 
            </div>
        """, unsafe_allow_html=True)
    st.write("\n")
    st.markdown("""
        <ol>
            <li style="margin-left: 14mm;">Réduire notre jeu de données d'un huitième</li>
            <li style="margin-left: 14mm;">Retirer les pixels les moins informatifs avec SelectPercentile(percentile = 90)</li>
            <li style="margin-left: 14mm;">Réduire les dimensions avec PCA(n_components = 0.9)</li>
            <li style="margin-left: 14mm;">Créer nos modèles de classification :
                <ul>
                    <li style="margin-left: 14mm;">SVM</li>
                    <li style="margin-left: 14mm;">Random Forest</li>
                </ul>
            </li>
            <li style="margin-left: 14mm;">Enfin prédire et obtenir les résultats</li>
        </ol>
        <br>""", unsafe_allow_html=True)
    st.write("\n")

    exp_eq4 = st.beta_expander("Illustration des traitements appliquées avant la création des modèles")

    options_eq4 = ["SelectPercentile", "PCA"]
    choix_eq4 = exp_eq4.radio("Choisissez une méthode :", options_eq4)

    image_eq4 = plot_eq4(choix_eq4, folderpath)
    exp_eq4.image(image_eq4, width=650)

    st.markdown("""<div style='text-align: justify'><br>
                Ci-dessous les résultats de nos modèles:
            <br><br></div>
        """, unsafe_allow_html=True)
    

    options_eq5 = ["SVM", "Random Forest"]
    choix_eq5 = st.radio("Choisissez un modèle :", options_eq5)

    image_eq5 = plot_eq5(choix_eq5, folderpath)
    st.image(image_eq5, width=750)


    st.markdown("""<div style='text-align: justify'>
            Nous pouvons constater que les classements de nos 2 modèles sont mauvais 
            et correspondent au hasard. De plus nous pouvons aussi remarquer un 
            overfitting sur les labels 0 et 1. <br>
            La méthode de réduction de dimension n'est pas concluante pour la 
            classification de nos images.
            Nous devons explorer de nouvelles voies. 
            </div>
        """, unsafe_allow_html=True)

    st.markdown("____")


#%% Fonction présentation du modèle Lenet
def lenet_page(folderpath):
    #%%% En tête :
    st.markdown("<h1 style='color:#FE9A1A;'>Architecture LeNet<br><br></h1>", unsafe_allow_html=True)
    
    st.write("L’architecture LeNet est introduit par LeCun et al. en 1998")

    
    imname1 = folderpath + "architecture.png"
    st.image(imname1, width=700)

    st.write("""<body style='text-align: justify'><br>
             LeNet est composé des couches de convolution suivis des couches de Pooling,
             puis des couches entièrement connectées, avec une dernière couche munie d'une fonction d'activation Softmax.</body>""", unsafe_allow_html=True)
    st.write("Voici le summary de l'architecture LeNet testée dans ce projet :")

    
    imname2 = folderpath + "summary_lenet.png"
    st.image(imname2, width=700)
    
    body1="""<body style='text-align: justify'><br>
    Nous avons utilisé la classe ImageDataGenerator pour augmenter le nombre de nos images 
        et éviter le surapprentissage. Nous avons utiliser la méthode flow_from_dataframe pour créer nos datasets de train et de test.
        Sur la figure ci-dessous, vous pouvez voir les plots de l’accuracy et la fonction de loss pour nos datasets de train et de test :<br><br></body>"""
    st.markdown(body1, unsafe_allow_html=True)
    

    imname3 = folderpath + "loss_accuracy_lenet.png"
    st.image(imname3, width=700)
    
    st.markdown("""<body style='text-align: justify'><br>
                Nous pouvons constater que nous n’avons pas une amélioration significative de notre accuracy ni 
                pour notre dataset de train, ni pour celui de test. La fonction de perte ne diminue pas dans les deux cas.</body>""",unsafe_allow_html=True)
    st.markdown("En affichant la matrice de confusion, nous constatons que le modèle prédit tous les labels en tant que label 0 :")
    st.write("")
    imname4 = folderpath + "matrice_confusion_lenet.png"
    st.image(imname4, width=500)
    st.subheader("Conclusion:")
    st.write("")
    st.markdown("Ces indicateurs nous prouvent que l’architecture LeNet n’est pas une architecture adaptée pour notre problématique de classification des champignons.")
    st.write("")

#%% Fonction de présentation des modeles transfer learning
def thirdpage_part2(df, df_clean, df_first, df1, folderpath):
#%%% Intro transfer learning    
    
    st.markdown("""<div style="color:#FE9A1A;font-size: 34px;font-weight: bold; line-height:45px;">
        Identification d'un modèle neuronal convolutionnel pertinent :
        </div>
        """, unsafe_allow_html=True)    


    st.markdown(""" <div style='text-align: justify'><br>
        Après avoir été confronté aux faibles performances des essais préalables
             (classification <b>SVC/Random Forest</b>, modèle <b>LeNet</b>), nous nous sommes tournés vers
            des modèles avec une architecture plus complexe et avons ainsi testé et optimisé
            une architecture par <i>transfer learning</i>.<br>
            <br>
    Le <i>transfer learning</i> est une approche qui consiste à transférer à un nouveau modèle, le savoir
    d’un modèle déjà entraîné. En l’occurrence, il s’agit de figer les poids du premier modèle et de transférer
    ces poids (issus de la résolution d’un problème précis) afin de répondre à une nouvelle problématique résolue
    par un nouveau modèle.
    De cette manière, le problème à résoudre bénéficie du résultat d’un entraînement préalable pour lequel
    nous n’aurions eu ni le temps, ni les ressources matérielles.
            </div><br>
        """, unsafe_allow_html=True)
        
    
#%%% Creation de la liste dans la sidebar    
    options_model = ["VGG16", "VGG19", "ResNet50", "Inception v3", "EfficientNetB1"]
    choix_model = st.sidebar.radio("Choisissez un modèle :", options_model)

#%%%% Présentation VGG16    
    if choix_model == options_model[0]:
        st.header("Modèle VGG16 :")
        
        if st.checkbox("Détails du modèle :"):
            st.subheader("Présentation :")        
            st.markdown(""" <div style='text-align: justify'>
                <a href="https://keras.io/api/applications/vgg/#vgg16-function">VGG16</a>
                est un réseau neuronal convolutif proposé par K. Simonyan et A. Zisserman de
                l’université d’Oxford dans un article publié : “Very Deep Convolutional Networks
                for Large-Scale Image Recognition”.<br>
                Ce dernier est nommé VGG16 car il contient tout simplement 16 couches profondes.
               """, unsafe_allow_html=True)
    
            imagename = folderpath + "modele_VGG16_archi.jpg"
            st.image(imagename)
            st.caption("Architecture du modèle VGG16 [(Source)](https://www.datacorner.fr/vgg-transfer-learning/)")    
            
            
            st.subheader("Résultats :")
            
            col1, col2 = st.beta_columns(2)
            col1.write("Historique d'entrainement :")
            imagename2 = folderpath + "20210707_training_history_modele_VGG16.png"
            col1.image(imagename2)
            
            col2.write("Matrice de confusion :")
            imagename3 = folderpath + "20210707_confusion_matrix_VGG16.png"
            col2.image(imagename3)
        
        st.text("Caractéristiques du modèle en fin d'entrainement : \n\n\
                loss: 1.1877 - acc: 0.5045 - val_loss: 0.9365 - val_acc: 0.6281")
        
        st.markdown(""" <div style='text-align: justify'>
            <br>Nous pouvons constater que le modèle ne classifie pas très efficacement
            nos images (mis à part pour la classe n°2). D’après notre exploration des
            images, cette classe était la classe la plus “facile” à prédire pour nos
            regards naïfs de non expérimentés.
            """, unsafe_allow_html=True)

#%%%% Présentation VGG19
    elif choix_model == options_model[1]:
        st.header("Modèle VGG19 :")
        
        if st.checkbox("Détails du modèle :"):
            st.subheader("Présentation :")
            st.markdown(""" <div style='text-align: justify'>
                <a href="https://keras.io/api/applications/vgg/#vgg19-function">VGG19</a>
                est un réseau neuronal convolutif également proposé par K. Simonyan et
                A. Zisserman de l’université d’Oxford dans le même article qui a présenté
                le modèle VGG16 (“Very Deep Convolutional Networks for Large-Scale Image
                Recognition”).<br>
                Ce dernier est nommé VGG19 car il contient tout simplement 19 couches profondes.<br><br>
               """, unsafe_allow_html=True)
            
            imagename = folderpath + "VGG_architecture.png"
            st.image(imagename)
            st.caption("Comparaison entre l’architecture des modèles VGG existants \
                       [(Source)](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)")    
    
            st.subheader("Résultats :")
            
            st.write("Historique d'entrainement :")
            imagename2 = folderpath + "20210707_training_history_modele_VGG19.png"
            st.image(imagename2)
        st.markdown(""" <div style='text-align: justify'>
            La fonction de perte reste très élevée et les performances du modèles
            sont également très modestes : dernière époque d’entraînement :
           """, unsafe_allow_html=True)
        
        st.text("loss: 1.1141 - acc: 0.5383 - val_loss: 0.9180 - val_acc: 0.6418")
        

#%%%% Présentation ResNet50        
    elif choix_model == options_model[2]:
        st.header("Modèle ResNet50 :")
        
        if st.checkbox("Détails du modèle :"):
            st.subheader("Présentation :")
            st.markdown(""" <div style='text-align: justify'>
                <a href="https://keras.io/api/applications/resnet/#resnet50-function">ResNet50</a>
                repose sur la mise au point d’une toute nouvelle architecture. Cette dernière 
                a notamment été primée lors de la compétition ImageNet en 2015 avec une avance 
                importante : diminution de moitié du taux d’erreur top-5 par rapport aux 
                compétiteurs (top-5 erreur sur le jeu de validation d’ImageNet : 5,25%).<br>
                Selon les architectures connues avant ResNet50 : plus un modèle comporte 
                de couches, plus on peut dire que le réseau est profond, meilleures sont 
                les performances. Néanmoins, tel que nous avons pu le constater entre VGG16 
                et VGG19, à partir d’une certaine profondeur, cette “règle” ne s’applique 
                plus et l’ajout de couches supplémentaires provoque l’effet inverse : une 
                dégradation des performances.<br>
                C’est pour pallier à ces défauts que les inventeurs de l’architecture ResNet
                ont inventé ce qu’ils ont appelé des connexions par saut, ainsi les différents
                niveaux sont reliés entre eux. L’optimisation de ce type de connexion serait
                alors plus facile que l’optimisation de réseaux plus “traditionnels”. 
                La notion de bloc résiduel est alors développée : l'entrée x est directement 
                ajoutée à la sortie du réseau. C’est la connexion qui lie l’entrée et la sortie 
                du réseau qui est nommée connexion par saut. Il a été suggéré que ce type 
                d’architecture fonctionne en réalité comme un ensemble de sous-réseaux plus 
                petits, et donc plus facile à entraîner.
               """, unsafe_allow_html=True)
            
            imagename = folderpath + "modele_resnet50_architecture_publi.jpg"
            st.image(imagename)
            st.caption("Architecture du modèle ResNet50 publiée par les inventeurs, \
                       en comparaison avec le modèle VGG19 (à gauche), un modèle \
                      “conventionnel” à 34 couches (au centre) et le modèle \
                      ResNet50 (à droite) illustrant les connexions par saut \
                      additionnant l’entrée à la sortie du bloc résiduel\
                       (source : Deep Residual Learning for Image Recognition, He et al 2016).")    
    
            st.subheader("Résultats :")
            
            st.write("Historique d'entrainement :")
            imagename2 = folderpath + "20210707_training_history_modele_ResNet50.png"
            st.image(imagename2)
        
        
        st.markdown(""" <div style='text-align: justify'>
            La fonction de perte reste très élevée et les performances du modèles
            sont également très modestes : dernière époque d’entraînement :
           """, unsafe_allow_html=True)
        
        st.text("loss: 1.0768 - acc: 0.5655 - val_loss: 0.9483 - val_acc: 0.6409")
                

#%%%% Présentation Inception v3
    elif choix_model == options_model[3]:
        st.header("Modèle Inception v3 :")
        
        if st.checkbox("Détails du modèle :"):
            st.subheader("Présentation :")
            st.markdown(""" <div style='text-align: justify'>
                <a href="https://keras.io/api/applications/inceptionv3/">Inception v3</a>
                est un réseau de neurones à convolution “classique” au premier abord en ce 
                sens puisqu’il utilise une succession de couches convolutives et de pooling
                dans sa phase d’extraction des données et un réseau de neurones dans sa phase 
                de classification. Néanmoins la particularité d’Inception réside dans la 
                présence de modules “inception”. Habituellement, plusieurs paramètres peuvent 
                être choisis pour une couche de convolution. En effet, par exemple, il faut 
                déterminer la taille du kernel, qui lui même déterminera les paramètres 
                extraits des images. Des tailles des 5x5, 3x3, 1x1 ou même 11x11 peuvent, 
                par convention, être utilisées. Ainsi, au lieu de devoir faire un choix, 
                les modules inception se composent comme suit: initialement, une première 
                convolution avec un kernel de taille 1x1 est réalisée. À partir de cette 
                convolution, deux nouvelles convolutions successives avec un kernel de 
                taille 5x5 puis de taille 3x3 sont réalisées. En parallèle, à partir de 
                la donnée d'entrée, un average pooling est réalisé. Finalement, l’ensemble 
                des données sont regroupées dans une seule matrice par concaténation17
                 (<a href="https://steemit.com/fr/@rerere/comment-fonctionne-un-reseau-de-neurones-inception-v3-4">source</a>)
                 .<br><br>
               """, unsafe_allow_html=True)
            
            imagename = folderpath + "modele_incpetionv3_architecture.png"
            st.image(imagename)
            st.caption("Architecture du modèle Inception v3 ([Source](https://cloud.google.com/tpu/docs/inception-v3-advanced?hl=fr))")    
    
            st.subheader("Résultats :")
            
            st.write("Historique d'entrainement :")
            imagename2 = folderpath + "20210707_training_history_modele_inceptionv3.png"
            st.image(imagename2)
        
        st.markdown(""" <div style='text-align: justify'>
            Ici nous pouvons constater que les performances sont probablement les 
            plus mauvaises de nos essais, la fonction de perte reste très élevée 
            et les performances du modèle sont également très modestes. Voici la 
            dernière époque d’entraînement :
           """, unsafe_allow_html=True)
        
        st.text("loss: 1.2934 - acc: 0.4551 - val_loss: 1.2739 - val_acc: 0.4801")
        
#%%%% Présentation EfficientNetB1
    elif choix_model == options_model[4]:
        st.header("Modèle EfficientNetB1 :")
        
        if st.checkbox("Détails du modèle :"):
            st.subheader("Présentation :")
            st.markdown(""" <div style='text-align: justify'>
                L’objectif de la mise en place des modèles EfficientNet (dont
                <a href="https://keras.io/api/applications/efficientnet/#efficientnetb1-function">EfficientNetB1</a>)
                a été de travailler sur la mise à l’échelle des modèles convolutifs. 
                Cette mise à l’échelle est habituellement une manière d’accroître les 
                performances des modèles. Néanmoins plusieurs paramètres peuvent être modifiés 
                et les processus menant aux gains de performances sont rarement expliqués. 
                Ainsi, les approches traditionnelles de mise à l’échelle se concentrent sur 
                <b>la largeur</b> du modèle (un modèle plus large est supposé capturer des 
                caractéristiques plus fines et être plus simple à entraîner), 
                sa <b>profondeur</b> (un modèle plus profond est également supposé être capable 
                de capter plus d’informations) ou encore la <b>résolution des images 
                d’entrée</b> (qui doit aussi permettre de capture des caractéristiques plus fines). 
                Cependant, bien que cette mise à l’échelle s’accompagne en général d’une 
                amélioration d’accuracy, elle s’accompagne également d’un coût de calcul 
                plus important (mesuré en FLOPS pour “floating point operations per second”). 
                Afin d’améliorer la mise à l’échelle, les auteurs proposent de travailler sur 
                la mise à l’échelle des 3 paramètres simultanément.<br><br>
               """, unsafe_allow_html=True)
            
            imagename0 = folderpath + "modele_efficientnetB1-01_architecture.png"
            imagename1 = folderpath + "modele_efficientnetB1-02_architecture.png"
            st.image(imagename0)
            st.image(imagename1)
            st.caption("Architecture du modèle EfficientNetB1 ([Source](https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142))")    
    
            st.subheader("Résultats :")
            
            st.write("Résumé du modèle entrainé :")
            imagename2 = folderpath + "20210707_modele_summary_effnetB1_initial.jpg"
            st.image(imagename2)
        
        
            st.markdown(""" <div style='text-align: justify'>
               <br><br>
               """, unsafe_allow_html=True)
            
            
            col1, col2 = st.beta_columns(2)
            col1.write("Classification report :")
            imagename3 = folderpath + "20210707_Classif_report_modele_efficientnetB1.jpg"
            col1.image(imagename3)
            
            col2.write("Matrice de confusion :")
            imagename4 = folderpath + "20210707_confusion_matrix_efficientnetB1.png"
            col2.image(imagename4)
        
        st.markdown(""" <div style='text-align: justify'>
            Nous avons donc obtenu une accuracy de 
            <a style="color:#64DB2D ; font-weight: bold;">71%</a>
            sur le set de test :
           """, unsafe_allow_html=True)
        
        st.text("loss: 0.1521 - accuracy: 0.9507 - val_loss: 1.0890 - val_accuracy: 0.7135")
        
        st.markdown(""" <div style='text-align: justify'>
           <br>
           <b>Ici, nous pouvons constater que les performances sont les meilleures de 
           nos essais (bien que le modèle souffre un peu de sur-apprentissage). Sur le set d'entraînement,
           la fonction de perte a été plutôt bien minimisée et les performances du modèle 
           sont raisonnables.<br>
           Nous avons donc choisi de continuer nos modélisations avec le modèle EfficientNetB1.</b>
           """, unsafe_allow_html=True)
    
        
        
        
        
        
        
        
        
        
        








    