import streamlit as st
import mlflow
import pandas as pd
from cpreprocess import preprocessing
from sklearn.metrics import accuracy_score
from ctrained_model import trained_model,compare_models



# Retrieve cols_drifted from session_state
drift_message = st.session_state.cols_drifted
#model_drift_message = st.session_state.model_drift
#best_run_id = st.session_state.best_runid
uploaded_new_data = st.session_state.uploaded_data

mlflow.set_tracking_uri("http://127.0.0.1:5046")
mlflow.set_experiment('sample')
client=mlflow.MlflowClient()


if "model_retrained" not in st.session_state:
    st.session_state.model_retrained = False


if drift_message != "" :
    st.write(drift_message)
    options = ['Yes', 'No']
    replace_model = st.radio("Would you like to replace the model?", options,index=None)
    if replace_model:
        if replace_model == 'Yes' and not st.session_state.model_retrained:


            run_id,version = trained_model(uploaded_new_data)
            st.session_state.best_runid = run_id
            st.session_state.model_retrained = True
            print('run_id of latest experiment :',run_id,'*'*500) 
            prod_model,staged_model=compare_models()
            if prod_model['accuracy_score_X_test'] <= staged_model['accuracy_score_X_test']:
                model_message = "The production accuracy has been low compared to the new model accuracy"
                st.write(model_message)
            else:
                model_message = ''
                st.write(model_message)

            st.write("The model utilized here has been trained on newly loaded data as you chose to replace the model.")
            version = max([i.version for i in client.get_registered_model('pr_model').latest_versions])
            client.transition_model_version_stage(name='pr_model',
                                                version=version,
                                                stage='Production', archive_existing_versions=True)
            client.set_registered_model_alias("pr_model", "champion", version=version)


        elif replace_model == 'No':
            st.write("The model utilized here has been trained on past data as you chose not to replace the model.")





            
    test_data = st.file_uploader(label='Upload your file', type='csv')
    if st.button("Predict"):
        if test_data is not None:
            test_data = pd.read_csv(test_data)
            # preprocess the df
            X_test, y_test, _, _ = preprocessing(test_data)
            model_name = 'pr_model'
            alias = 'champion'
            model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
            pred = model.predict(X_test)
            pred = (pred > 0.5).astype(int)
            print('y_test:', y_test)
            print('pred:', pred)
            accuracy = accuracy_score(y_test, pred)
            st.write(f'The accuracy score is {accuracy}')
        else:
            st.write("Please upload a file.")
   
    
else:
    print("No drift message or model drift message.")
