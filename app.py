# Importing necessary libraries
import streamlit as st
import threading,subprocess

import os
import mlflow
from dotenv import load_dotenv
load_dotenv()



def main():

    
    def start_mlflow():
        subprocess.run(["mlflow", "server", "--backend-store-uri", "sqlite:///sample.db","--default-artifact-root", "wasbs://samplecon@vinstore1234.blob.core.windows.net/", "--host", "0.0.0.0", "--port", "5046"])
    
    mlflow_thread = threading.Thread(target=start_mlflow)
    mlflow_thread.start()


    mlflow.set_tracking_uri("http://127.0.0.1:5046")
    mlflow.set_experiment('sample')
    st.title('Prediction of Claims Risk Data')


    if st.button('Single Prediction'):
        st.switch_page('pages/single_prediction.py')
    if st.button('Batch Prediction'):
        st.switch_page('pages/batch_prediction.py')


if __name__ =='__main__':
    
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
    main() 



