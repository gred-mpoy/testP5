import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow
import tensorflow_hub as hub

# Charge le modèle et définit une fonction pour prédire des tags à partir d'une question
model = pickle.load(open('trained_model.pkl','rb'))
multilab_bin = pickle.load(open('multilab_bin2','rb'))

def predict_tags(quest):
  embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  quest= quest.split("'")
  features=embed(tensorflow.constant(quest))

  tags = model.predict(features)
  predicted_tags = multilab_bin.inverse_transform(tags)
  return predicted_tags

# Demande à l'utilisateur de saisir une question
question = st.sidebar.text_input("Saisissez votre question :")

# Ajoute un bouton "Envoyer" qui appelle la fonction predict_tags lorsqu'il est cliqué
if st.sidebar.button("Envoyer"):
  predicted_tags = predict_tags(question)
  st.write("Tags prédits :", predicted_tags)