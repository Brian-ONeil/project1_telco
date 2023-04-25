#######IMPORTS

import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split


#######FUNCTIONS



def prep_telco(telco_df):
    '''
    This function will clean the the telco dataset
    '''
    
    dummy_df = pd.get_dummies(telco_df[['multiple_lines',
                                 'online_security',
                                 'online_backup',
                                 'device_protection', 
                                 'tech_support',
                                 'streaming_tv',
                                 'streaming_movies', 
                                 'contract_type', 
                                 'internet_service_type',
                                 'payment_type']],
                              drop_first=True)

    telco_df = pd.concat([telco_df, dummy_df], axis=1)
    return telco_df

def split_data(df, stratify_col):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2,
                                   random_state=123,
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                    test_size=.25,
                                    random_state=123,
                                    stratify=train[stratify_col])
    return train, validate, test