#!/usr/bin/env python
# coding: utf-8

# ## Creating a Dashboard

# In[1]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# In[2]:


# Cargamos el modelo
filename = 'KNNRisk_model_13102020'
model = load_model(filename)
# model = pickle.load(open(filename, 'rb'))


# Now, let us write a function to predict the output for different inputs we get through the web interface.

# In[3]:


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


# The inputs can either be online where users enter the values in fields provided and get the predictions, or it can be through a CSV file. We will implement both these features. (When creating fields, it has to be in the same order as the training data). 

# In[5]:


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
        MONTO_PRESTAMO =st.number_input('Monto del Préstamo',min_value=0.00, max_value=100000.00, value=0.00)
        MORO_PROM = st.number_input('Morosidad promedio (días)', min_value=-30.00, max_value=150.00, value=0.00)
        PROMEDIO_INGRESOS = st.number_input('Promedio ingresos ($)',  min_value=0.00, max_value=2100000.00, value=0.00)
        MESES_DEFAULTS = st.number_input('Meses en moratoria',  min_value=0.00, max_value=24.00, value=0.00)
        PORC_CUMP = st.number_input('Porcentaje de cumplimiento',  min_value=0.00, max_value=1.00, value=0.00)
        EMI = st.number_input('EMI', min_value=0.00, max_value=5.00, value=0.00)
        BALANCE_INGRESOS = st.number_input('Balance ingresos ($)',  min_value=-700.00, max_value=2100000.00, value=0.00)
       
        output=""
        input_dict={'Monto del Préstamo':MONTO_PRESTAMO,
                    'Morosidad promedio (días)':MORO_PROM,
                    'Promedio ingresos ($)':PROMEDIO_INGRESOS,
                    'Meses en moratoria':MESES_DEFAULTS,
                    'Porcentaje de cumplimiento':PORC_CUMP,
                    'EMI':EMI,
                    'Balance ingresos ($)': BALANCE_INGRESOS
                    }
                    
                
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
            
# run()

if __name__ == "__main__":
    run()


# In[4]:


# Once we have implemented this, it is time to run the application. The command to do this is 
# streamlit run “applicationname.py”


# In[ ]:


# MONTO_PRESTAMO','MORO_PROM','PROMEDIO INGRESOS','PORC_CUMP','EMI','BALANCE_INGRESOS'
# EMI = (MONTO_PRESTAMO/PROMEDIO INGRESOS)*MESES DEFAULT (Cantidad de deuda no pagada por los meses en default)
# EMI_MORO = EMI + MORO_PROM * MESES DEFAULT
# PORC_CUMP es el pocentaje de meses que efectivamente pagó en los últimos 6 meses (1=100%, 0.33 = 33%, etc)
# BALANCE_INGRESOS = PROMEDIO INGRESOS - (PROMEDIO INGRESOS*EMI)

