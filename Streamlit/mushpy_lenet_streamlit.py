import streamlit as st
#%% fuction for the second page
def lenet_page():

    #%%% En tête :
    st.markdown("<h1 style='text-align: center'>Architecture LeNet</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("L’architecture LeNet est introduit par LeCun et al. en 1998" )
    st.write("")
    st.image("architecture.png", width=700)
    st.write("")
    st.write("""<body style='text-align: justify'>LeNet est composé des couches de convolution suivis des couches de Pooling,
             puis des couches entièrement connectées, avec une dernière couche munie d'une fonction d'activation Softmax.</body>""", unsafe_allow_html=True)
    st.write("Voici le summary de l'architecture LeNet testée dans ce projet:")
    st.write("")
    st.image("summary_lenet.png", width=700)
    st.write("")
    body1="""<body style='text-align: justify'>Nous avons utilisé la classe ImageDataGenerator pour augmenter le nombre de nos images 
        et éviter le surapprentissage. Nous avons utiliser la méthode flow_from_dataframe pour créer nos datasets de train et de test.
        Sur la figure ci-dessous, vous pouvez voir les plots de l’accuracy et la fonction de loss pour nos datasets de train et de test :</body>"""
    st.markdown(body1, unsafe_allow_html=True)
    st.write("")
    st.write()
    st.image("loss_accuracy_lenet.png", width=700)
    st.write("""<body style='text-align: justify'>Nous pouvons constater que nous n’avons pas une amélioration significative de notre accuracy ni 
                pour notre dataset de train, ni pour celui de test. La fonction de perte ne diminue pas dans les deux cas.</body>""",unsafe_allow_html=True)
    st.markdown("En affichant la matrice de confusion, nous constatons que le modèle prédit tous les labels en tant que label 0 :")
    st.write("")
    st.image("matrice_confusion_lenet.png", width=500)
    st.subheader("Conclusion:")
    st.write("")
    st.markdown("Ces indicateurs nous prouvent que l’architecture LeNet n’est pas une architecture adaptée pour notre problématique de classification des champignons.")
    st.write("")



# %%
