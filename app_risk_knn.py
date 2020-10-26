#!/usr/bin/env python
# coding: utf-8

# ## Aplicación para captura de datos en formulario par enviarlos al modelo de evaluación de riesgo de Default

# In[5]:


#import libraries
#import pandas as pd
import numpy as np
from flask import Flask, render_template,request
import pickle


# In[6]:


#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('KNNRisk_model_13102020.pkl', 'rb'))
#cols = ['MONTO_PRESTAMO','MORO_PROM','PROMEDIO_INGRESOS','MESES_DEFAULTS', 'PORC_CUMP', 'EMI','BALANCE_INGRESOS']


# In[7]:


# Define the app route for the default page of the web-app :
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# We create a new app route (‘/predict’) that reads the input from our ‘index.html’ form and on clicking the predict button, outputs the result using render_template.

# In[8]:


#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    
    # For rendering results on HTML GUI
     
    item = [x for x in request.form.values()]
    # final = np.array(item)
    # data_form = pd.DataFrame([final], columns=cols)
    final = [np.array(item)]
         
    prediction = model.predict(final)
    output = prediction[0]
    # predictions_df = predict_model(estimator=model, data=data_form)
    # predictions = predictions_df['Label'][0]
    
    return render_template('index.html', prediction_text='La predicción es:  {}'.format(output))


# In[9]:


#Starting the Flask Server
if __name__ == "__main__":
    app.debug = True
    app.run()





