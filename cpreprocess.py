from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def preprocessing(df):
    '''
    it is preprocessing the data.
    '''
    df_X = df.drop(columns=['HAS_CLAIMED_IN_NEXT_12_MONTHS'])
    df_y = df['HAS_CLAIMED_IN_NEXT_12_MONTHS']

    # Preprocess categorical data
    df_cat_data = df_X.select_dtypes(include=['object'])
    enc = LabelEncoder()
    df_cat = df_cat_data.apply(enc.fit_transform)

    # Preprocess numerical data
    df_num_data = df_X.select_dtypes(include=['int64', 'int32', 'float64'])
    scaler = StandardScaler()
    df_num = scaler.fit_transform(df_num_data)
    df_num = pd.DataFrame(df_num, columns=df_num_data.columns, index=df_num_data.index)

    # Concatenate features
    df_X = pd.concat([df_cat, df_num], axis=1)
    print('Data preprocessing done.')

    return df_X, df_y, df_cat_data, df_num_data


