#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:31:39 2021

@author: Adrien
"""

#%% Imports
import streamlit as st
import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model

#%% Load data

@st.cache(allow_output_mutation=True)
def load_datas(global_data, clean_data, data_first_model, final_data_set):
    df = pd.read_csv(global_data)
    df_clean = pd.read_csv(clean_data)
    df_first = pd.read_csv(data_first_model)
    df1 = pd.read_csv(final_data_set)
    
    return df, df_clean, df_first, df1

#%%
@st.cache(allow_output_mutation=True)
def nan_analysis(df, df_clean):
    sp_nan = df["species_1"].isna().sum()
    genre_nan = df["genus_2"].isna().sum()
    fam_nan = df["family_3"].isna().sum()
    sp_nan2 = df_clean["species_1"].isna().sum()
    genre_nan2 = df_clean["genus_2"].isna().sum()
    fam_nan2 = df_clean["family_3"].isna().sum()
    data_nan = {'Nombre de NaN (global)':[sp_nan, genre_nan, fam_nan],
                'Nombre de NaN (thumbnail = 1)':[sp_nan2, genre_nan2, fam_nan2]}
    
    data_nan2 = pd.DataFrame.from_dict(data_nan, orient='index', columns=["Espèce", "Genre", "Famille"])
    return data_nan2

#%%
#@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None}, allow_output_mutation=True)
def plot_eq(choix_eq, folderpath):
    
    options_eq = ["Phylum", "Classe", "Ordre"]
    
    if choix_eq == options_eq[0]:
        image_eq = folderpath + "phylum_eq.png"
    elif choix_eq == options_eq[1]:
        image_eq = folderpath + "classe_eq.png"   
    elif choix_eq == options_eq[2]:
        image_eq = folderpath + "ordre_eq.png"
        
    return image_eq


#%%
#@st.cache(allow_output_mutation=True)
def plot_eq2(choix_eq2, folderpath):
    
    options_eq2 = ["Famille", "Genre"]
    
    if choix_eq2 == options_eq2[0]:
        image_eq2 = folderpath + "famille_eq.png"
    elif choix_eq2 == options_eq2[1]:
        image_eq2 = folderpath + "genre_eq.png"  
    
    return image_eq2

def plot_eq3(choix_eq3, folderpath):
    options_eq3 = ["Isomap", "PCA"]

    if choix_eq3 == options_eq3[0]:
        image_eq3 = folderpath + "EDA_isomap.png"
    elif choix_eq3 == options_eq3[1]:
        image_eq3 = folderpath + "EDA_PCA.png"

    return image_eq3

def plot_eq4(choix_eq4, folderpath):
    options_eq4 = ["SelectPercentile", "PCA"]

    if choix_eq4 == options_eq4[0]:
        image_eq4 = folderpath + "Model_SelectPercentile.png"
    elif choix_eq4 == options_eq4[1]:
        image_eq4 = folderpath + "Model_PCA.png"

    return image_eq4

def plot_eq5(choix_eq5, folderpath):
    options_eq5 = ["SVM", "Random Forest"]

    if choix_eq5 == options_eq5[0]:
        image_eq5 = folderpath + "Model_SVM.png"
    elif choix_eq5 == options_eq5[1]:
        image_eq5 = folderpath + "Model_RandomForest.png"

    return image_eq5

#%% Loading models
@st.cache(allow_output_mutation=True)
def loading_models(model_fam_path, model_label0_path, model_label1_path, model_label2_path, model_label3_path, model_label4_path):    
    model_fam = load_model(model_fam_path)
    model_label0 = load_model(model_label0_path)
    model_label1 = load_model(model_label1_path)
    model_label2 = load_model(model_label2_path)
    model_label3 = load_model(model_label3_path)
    model_label4 = load_model(model_label4_path)
    return model_fam, model_label0, model_label1, model_label2, model_label3, model_label4


#%% Functions for Grad-CAM

def softmax(vector):
    e = np.exp(vector)
    vector_proba = e / e.sum()
    final = round(max(vector_proba)*100, 2)
    return vector_proba, final

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, inner_model, last_conv_layer_name, pred_index=None):
    if inner_model == None:
        inner_model = model
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(inputs=[inner_model.inputs],
                      outputs=[inner_model.get_layer(last_conv_layer_name).output,
                               inner_model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        (last_conv_layer_output, preds) = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
#    pooled_grads = tf.reduce_mean(grads, axis=-1)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, beta=1):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img2 = img
    #display(img2) ## uncomment if you want the funtion to directly display the image
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    jet_heatmap2 = keras.preprocessing.image.array_to_img(jet_heatmap)
    #display(jet_heatmap2) ## uncomment if you want the funtion to directly display the image

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img * beta
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))
    
    
    #display(superimposed_img) ## uncomment if you want the funtion to directly display the image
    return img2, jet_heatmap2, superimposed_img


#%% Make the prediction and plot the result :

def prediction(filename_imagetest, model_fam, model_label0, model_label1, model_label2, model_label3, model_label4):
    #%%% Define the family/genus names of mushrooms    
        family_names = ['Inocybaceae','Omphalotaceae','Fomitopsidaceae','Physalacriaceae','Marasmiaceae']
    
        genus_names_label0 = ['Inocybe', 'Crepidotus', 'Simocybe', 'Flammulaster']
        genus_names_label1 = ['Gymnopus', 'Omphalotus', 'Rhodocollybia', 'Marasmiellus']
        genus_names_label2 = ['Fomitopsis', 'Laetiporus', 'Phaeolus', 'Postia', 'Ischnoderma', 'Piptoporus', 'Daedalea', 'Antrodia']
        genus_names_label3 = ['Armillaria', 'Hymenopellis', 'Flammulina', 'Strobilurus', 'Oudemansiella', 'Cyptotrama']
        genus_names_label4 = ['Marasmius', 'Megacollybia', 'Gerronema', 'Tetrapyrgos', 'Atheniella', 'Clitocybula']
        
    #%%% Prediction    
    #%%%% Define few parameter important as image size, the name of the las convolution layer...    
        img_size = (256, 256)
        last_conv_layer_name = "top_conv"
        inner_model_fam = model_fam.get_layer("efficientnetb1")
        preprocess_input = keras.applications.efficientnet.preprocess_input
    #%%%% Prepare the image to be predicted    
        img_array = preprocess_input(get_img_array(filename_imagetest, size=img_size))
        
    #%%%%% Remove last layer's softmax    
        model_fam.layers[-1].activation = None
    
    #%%%% What the top predicted class is
        preds = model_fam.predict(img_array)
        predict = tf.argmax(preds, axis = 1).numpy()
        final_prob = round(max(preds[0])*100, 2)
        
        vector_probability , final_probability = softmax(preds[0])
        class_predict = np.argmax(vector_probability)
        
        
    #%%%% Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model_fam, inner_model_fam, last_conv_layer_name, pred_index=predict[0])
        
        
    #%%%% Indicate the predicted family number
        
        fam_number = predict[0]
        
        if fam_number == 0:
            model_genus = model_label0
            genus_name = genus_names_label0
        elif fam_number == 1:
            model_genus = model_label1
            genus_name = genus_names_label1
        elif fam_number == 2:
            model_genus = model_label2
            genus_name = genus_names_label2
        elif fam_number == 3:
            model_genus = model_label3
            genus_name = genus_names_label3
        elif fam_number == 4:
            model_genus = model_label4
            genus_name = genus_names_label4
        
        inner_model_genus = model_genus.get_layer("efficientnetb1")
            
    #%%%% Do the prediction for genus (as before for family)
        model_genus.layers[-1].activation = None
        
        preds_genus = model_genus.predict(img_array)
        predict_genus = tf.argmax(preds_genus, axis = 1).numpy()
        final_prob_genus = round(max(preds_genus[0])*100, 2)
        
        vector_probability_genus , final_probability_genus = softmax(preds_genus[0])
        class_predict_genus = np.argmax(vector_probability_genus)
    
    #%%% Plot the results !
        title1 = "La famille predite avec {}% de certitude est la famille : \n{} (classe {})" .format(final_probability, family_names[class_predict], class_predict)
        
        title2 = "Le genre predit avec {}% de certitude est le genre : \n{} (classe {})" .format(final_probability_genus, genus_name[class_predict_genus], class_predict_genus)
    
        img, jet_heatmap2, superimposed_img = save_and_display_gradcam(filename_imagetest, heatmap)
        fig9 = plt.figure(figsize=(8,13))
        plt.subplots_adjust(wspace=0, hspace=0.02)
        plt.subplot(311)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(title1 + "\n\n" + title2)
        plt.subplot(312)
        plt.imshow(jet_heatmap2)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(313)
        plt.imshow(superimposed_img);
        plt.xticks([])
        plt.yticks([]);
        #st.pyplot(fig9)
        return fig9


#%% Fonction pour indiquer le process image data generator

def imagedata(folderpath):
    
    st.markdown("""<div>
                <b><u>Augmentation des images :</b></u>
                    </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: justify'>
        La librairie Keras permet d'augmenter les images lors de l'entrainement d'un modèle. Cette augmentation
        permet notamment de lutter contre le surapprentissage. En effet, à chaque étape de l'apprentissage,
        le modèle est chargé avec une image "différente" dans la mesure où l'image originale subit des 
        transformations aléatoires. Les images sont certes transformées, mais pas complètement changées. Cela
        permet en particulier d'éviter au modèle de toujours se focaliser sur les même détails. Il s'agit
        également d'une stratégie qui permet de compenser des jeux de données non-équilibrés<br>
        La classe <b>ImageDataGenerator</b> est utilisée à cet effet.<br>
        <br>
        Un ensemble de techniques sont disponibles, parmis lesquelles :
        <br>
        <ul>
            <li style="margin-left: 20mm;"><i>width_shift_range</i> et <i>height_shift_range décalent les images</i></li>
            <li style="margin-left: 20mm;"><i>horizontal_flip</i> et <i>vertical_flip retournent les images</i></li>
            <li style="margin-left: 20mm;"><i>rotation_range</i> pour exercer une rotation sur les images</li>
            <li style="margin-left: 20mm;"><i>zoom_range</i> fait un zoom sur les images</li>
        </ul>
        <br>
        <div>
        """, unsafe_allow_html=True)

    st.markdown("""
        Voici un exemple d'augmentation d'une image, 
        prenons une image de morille :
        """, unsafe_allow_html=True)

    imagepath = folderpath + "imagedatagenerator_original.png"
    st.image(imagepath)

    st.markdown("""
        <br>
        Appliquons les 4 transformations suivantes via <b>ImageDataGenerator</b> :
        """, unsafe_allow_html=True)

    imagepath = folderpath + "imagedatagenerator_transformations.png"
    st.image(imagepath)


    st.markdown("""
        <br>
        Avec les 4 transformations précédentes, l'image ressemble finalement à :
        """, unsafe_allow_html=True)

    imagepath = folderpath + "imagedatagenerator_final.png"
    st.image(imagepath)
    
    st.markdown("""
        <br><br>
        """, unsafe_allow_html=True)


#%%





