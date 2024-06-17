# import pandas as pd
# import numpy as np
import streamlit as st
# from ctrained_model import trained_model
# import mlflow

st.header('Single prediction', divider='rainbow')

# enc = trained_model()
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("sample_iris")
# model=mlflow.pyfunc.load_model('runs:/e2812b25a8af4ee886a68242978b39c7/')

# sl = st.number_input(label='sepal_length')
# sw = st.number_input(label='sepal_width')
# pl = st.number_input(label='petal_length')
# pw = st.number_input(label='petal_width')
# values = np.array([sl,sw,pl,pw]).reshape(1,-1)
# if st.button('Predict'):
#     pred = model.predict(values)
#     species = enc.inverse_transform(pred)


#     st.write(f'the flower predicted is {species}')
pass
if st.button('Back to Home'):
    st.switch_page('app.py')