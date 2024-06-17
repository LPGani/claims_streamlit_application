import pandas as pd
import streamlit as st
#from trained_model import trained_model,eval_metrics
import webbrowser
from numerical_drift import numerical_data_drift



st.header('Batch prediction', divider='rainbow')

#model,enc,y_test = trained_model()
new_data = st.file_uploader(label='Upload your file',type='csv')
if new_data is not None:
    # databse connection
    # sql query - retrieve either latest 100 records or date range
    uploaded_new_data = pd.read_csv(new_data)
    st.session_state.uploaded_data = uploaded_new_data

    
    if st.button('Check data drift'):
        st.write("checking for data drift, might take time...")
        drift,drift_message = numerical_data_drift(new_data=uploaded_new_data)
        message = 'There is no data drift detected.'
        if len(drift)>1:
            #cols_drifted = drift_message
            st.session_state.cols_drifted = drift_message
        else:
            st.session_state.cols_drifted = message

        
        

        st.switch_page('pages/drift.py')


        
# Provide link to MLflow UI
if st.button('View MLflow UI'):
    
    
    # st.markdown("[View MLflow UI](http://127.0.0.1:5000)")
    mlflow_ui_url = 'http://127.0.0.1:5046'
    webbrowser.open_new_tab(mlflow_ui_url)






        

if st.button('Back to Home'):
    st.switch_page('app.py')


