#######IMPORTS

import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split


#######FUNCTIONS



def prep_telco(telco_df):
    '''
    This function will clean the the telco dataset, create dummies for four categories, concat, drop unneeded files, then strip to lower case.
    '''
    
    dummy_df = pd.get_dummies(telco_df[['tech_support',
                                 'contract_type', 
                                 'internet_service_type',
                                 'payment_type']],
                              drop_first=True)

    telco_df = pd.concat([telco_df, dummy_df], axis=1)
    telco_df = telco_df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'gender', 'senior_citizen',
                                      'partner', 'dependents', 'tenure', 'phone_service', 'multiple_lines', 'online_security',
                                      'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
                                      'paperless_billing', 'total_charges', 'contract_type', 'internet_service_type', 'payment_type'])
    telco_df = telco_df.rename(columns=str.lower)
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