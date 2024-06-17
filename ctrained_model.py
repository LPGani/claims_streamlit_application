import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.metrics import accuracy_score

from cpreprocess import preprocessing
import mlflow
# from mlflow.tracking import MlflowClient

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope



def trained_model(df):
    # Preprocess data within the function
    preprocessed_df_X, preprocessed_df_y, _, _ = preprocessing(df.copy())
    X_train,X_val,y_train,y_val = train_test_split(preprocessed_df_X,preprocessed_df_y,test_size=0.25,random_state=42)

    # Setting the mlflow db and experiment name   
    mlflow.set_tracking_uri("sqlite:///sample.db")
    mlflow.set_experiment("sample")
    client=mlflow.MlflowClient()

    print('Deleting previous runs')

    for i in client.search_runs(mlflow.get_experiment_by_name("sample").experiment_id):
        id=i.to_dictionary()['info']['run_id']
        mlflow.delete_run(id)

    # Assuming you have prepared your data for classification
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',  # Change to appropriate classification objective
        'seed': 42
    }

    def objective(params):
        with mlflow.start_run():

            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=30
            )
            y_pred = booster.predict(valid)
            
            # Example: Calculate accuracy for binary classification
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred_binary)
            mlflow.log_metric("accuracy", accuracy)

        return {'loss': 1 - accuracy, 'status': STATUS_OK}  # Minimize 1 - accuracy

    print('Training initiated')
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials()
    )

    run_id=[]
    accuracy=[]
    parameters=[]
    run_name=[]

    print('Tracking runs')
    runs=pd.DataFrame()

    for i in client.search_runs(experiment_ids=[client.get_experiment_by_name('sample').experiment_id]):
        run_name.append(i.to_dictionary()['data']['tags']['mlflow.runName'])
        run_id.append(i.to_dictionary()['info']['run_id'])
        accuracy.append(i.to_dictionary()['data']['metrics']['accuracy'])
        parameters.append(i.to_dictionary()['data']['params'])

    runs['run_name']=run_name
    runs['run_id']=run_id
    runs['accuracy']=accuracy
    runs['parameters']=parameters

    best_model=runs.sort_values('accuracy').tail(1)

    best_params=best_model['parameters'].values[0]
    

    print('Parameter tuning done and best parameters are',best_params)

    print('Model training with best parameters')

    print("training and logging artifacts of the best model")

    with mlflow.start_run(experiment_id=1,run_name=best_model['run_name'].values[0]) as run:
        mlflow.autolog()
        X_train,X_val,y_train,y_val=X_train,X_val,y_train,y_val
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=30
            )

        y_pred = booster.predict(valid)

        # Example: Calculate accuracy for binary classification
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy=accuracy_score(y_pred=y_pred_binary,y_true=y_val)
        mlflow.log_metric('accuracy_score_X_test',accuracy)

        model_uri="runs:/{}/model".format(run.info.run_id)

        print('\nModel retraining done with run_id:',run.info.run_id)
        print('RUNSSSSSS::::::::::',runs,'#'*500)
        best_runid = run.info.run_id
        print(best_runid)
        print('Model uri:',model_uri)
        print('run uuid\n',run.info.run_uuid)

        print('\nRegister Best model\n')
        mlflow.register_model(name='pr_model',model_uri=model_uri)

        version=max([i.version for i in client.get_registered_model('pr_model').latest_versions])
        client.transition_model_version_stage(name='pr_model',
                                              version=version,
                                              stage='Staging')
        client.set_registered_model_alias("pr_model", "challenger", version=version)


    return best_runid, version
    



def compare_models():
    mlflow.set_tracking_uri("sqlite:///sample.db")
    
    print('Comparing runs')

    client=mlflow.MlflowClient()

    prod_model={'Stage':'Production'}
    staged_model={'Stage':'Staging'}

    for i in client.get_registered_model('pr_model').latest_versions:
        if i.current_stage=='Production':
            prod_model['run_id']=i.run_id
            metric=client.get_metric_history(run_id=i.run_id,key='accuracy_score_X_test')[0].value
            prod_model['accuracy_score_X_test']=metric

        elif i.current_stage=='Staging':
            staged_model['run_id']=i.run_id
            metric=client.get_metric_history(run_id=i.run_id,key='accuracy_score_X_test')[0].value
            staged_model['accuracy_score_X_test']=metric

    print('Production model:',prod_model)
    print('Staged model:',staged_model)

    return prod_model, staged_model


