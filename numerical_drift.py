import requests
from scipy import stats
import pandas as pd
import os
#from dotenv import load_dotenv
from cpreprocess import preprocessing

# def configure():
#     '''
#     Configures the env variables.
#     '''
#     load_dotenv()


# Defining the function for sending messages to teams.
def send_teams_message(message):
    '''
    Sends alert messages to teams dummy_mlops channel using Incoming Webhooks from Teams.
    '''
    webhook_url = os.getenv('teams_webhook_url')
    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 200:
        print("Message Alert sent successfully to Teams.")
    else:
        print(f"Failed to send message to Teams. Status code: {response.status_code}")





# Defining the function for numerical data drift.
def numerical_data_drift(new_data):
    '''
    Calculates the numerical data drift for past train data and current train data
    '''

    # Taking the past data numerical data
    past_data = pd.read_csv(r'Past_data_train.csv')
    _, _, _, pd_num_data = preprocessing(past_data)

    cur_data = new_data
    _, _, _, cd_num_data = preprocessing(cur_data)


    p_value = 0.05

    rejected = 0

    cols_rejected = []
    for col in pd_num_data.columns:

        test = stats.ks_2samp(pd_num_data[col], cd_num_data[col])

        if test[1] < p_value:
            rejected += 1
            print("Column rejected", col)
            cols_rejected.append(col)
            

    print("We rejected",rejected,"columns in total")
    # Sending message alerts to Teams dummy_mlops channel...
    message = f'The Columns rejected due to data drift are: {cols_rejected}'
    #send_teams_message(message)

    return cols_rejected,message

