#!/usr/bin/env python
# coding: utf-8

# ## Creating a Dashboard

# In[1]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('LR_Risk_Model_23092020')


# Now, let us write a function to predict the output for different inputs we get through the web interface.

# In[2]:


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


# The inputs can either be online where users enter the values in fields provided and get the predictions, or it can be through a CSV file. We will implement both these features. (When creating fields, it has to be in the same order as the training data). 

# In[3]:


def run():
    from PIL import Image
    image = Image.open('Loan_Prediction_Problem.png')
    image_logo = Image.open('logowaku.jpg')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
    "Como quiere predecir?",
    ("Online", "Batch"))
    st.sidebar.info('Esta App se encarga de predecir la probabilidad de Default de una persona')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_logo)
    st.title("Predicción de Probabilidad de Default")
    if add_selectbox == 'Online':
        Credit_History = st.selectbox('Credit_History', ['1', '0'])
        Total_Income =st.number_input('Total_Income',min_value=1, max_value=500000, value=1)
        Total_Income_log = st.number_input('Total_Income_log', min_value=1.00, max_value=100.0, value=1.00)
        EMI = st.number_input('EMI', min_value=0.01, max_value=500.00, value=0.01)
        Balance_Income = st.number_input('Balance_Income',  min_value=2000, max_value=500000, value=2000)
       
        output=""
        input_dict={'Credit_History':Credit_History,'Total_Income':Total_Income,'Total_Income_log':Total_Income_log,
                    'EMI':EMI,'Balance_Income': Balance_Income}
        
        input_df = pd.DataFrame([input_dict])
        if st.button("Predecir"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('El resultado es: {}'.format(output))
        
    # Let us now create an option for uploading CSV files as input 
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Cargue el archivo CSV", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

run()
# In[4]:


# Once we have implemented this, it is time to run the application. The command to do this is 
# streamlit run “applicationname.py”


# In[ ]:




