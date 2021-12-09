#preprocessing
import pandas as pd
import numpy as np
import functools

def get_data(rr_y1,rr_y2,rr_y3,rr_y4,tr_y1,tr_y2,tr_y3,tr_y4,rp,bool_tr2014):
    # get_data concatenates all the input data into one dataframe
    # dataframe will consist of teams in every team record dataset
    # rr_y# is the recruiting ranking of the team # years from the test set (eg for testing on 2018, y1 is 2014)
    # tr_y# is the team record # years from the test set (eg for testing on 2018, y1 is 2014)
    # rp is returning production of the team in the test year
    df_rr1 = pd.read_csv(rr_y1).drop(columns='year')
    df_rr2 = pd.read_csv(rr_y2).drop(columns='year')
    df_rr3 = pd.read_csv(rr_y3).drop(columns='year')
    df_rr4 = pd.read_csv(rr_y4).drop(columns='year')
    if bool_tr2014 == True:
        df_tr1 = pd.read_csv(tr_y1).drop(columns='year')
    else:
        df_tr1 = pd.read_csv(tr_y1)
    df_tr2 = pd.read_csv(tr_y2)
    df_tr3 = pd.read_csv(tr_y3)
    df_tr4 = pd.read_csv(tr_y4)
    df_rp = pd.read_csv(rp).drop(columns='season')
    data_frames = [df_rr1,df_rr2,df_rr3,df_rr4,df_tr1,df_tr2,df_tr3,df_tr4,df_rp]
    df_merged_r = functools.reduce(lambda  left,right: pd.merge(left,right,on=['team'],how='inner'), data_frames)
    #k = np.arange(len(data_frames)).astype(str)
    #df_merged_c = pd.concat([x.set_index('team') for x in data_frames], axis=1, join='inner')
    #print(df_merged_r.head)
    #df_merged_r.to_csv('data/merged_concat_no_year_with_teams.csv', header=True)
    #df_merged_r.to_csv(r'C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\merged_reduce.csv',header=True)
    #df_merged_c.to_csv(r'C:\Users\natha\OneDrive\Documents\GitHub\cs1470\BROWN-DL-FINAL-EM_NK\data\merged_concat_no_year.csv',header=True)
    #df_merged_c.to_csv('data/merged_concat_no_year.csv',header=True)
    return df_merged_r
    pass

def get_labels(exp_wins, tt, ppa, year):
    # this function concatenates all data necessary for label creation into one dataframe
    # exp_wins are the expected wins from either the 2018 or 2019 season
    # tt are the team_talent rankings from either the 2018 or 2019 season
    # ppa are the predicted points added metrics from the 2018 or 2019 season
    df_exp_wins_1 = pd.read_csv(exp_wins).drop(columns='year')
    df_tt_1 = pd.read_csv(tt).drop(columns='year')
    df_ppa_1 = pd.read_csv(ppa).drop(columns='conference')
    data_frames = [df_exp_wins_1, df_tt_1,df_ppa_1]
    df_merged_r = functools.reduce(lambda left, right: pd.merge(left, right, on=['team'], how='inner'), data_frames)
    #print(df_merged_r.head)
    #if year == 2019:
        #df_merged_r.to_csv('data/labels_test_merged_concat.csv',header=True)
    #else:
        #df_merged_r.to_csv('data/labels_train_merged_concat.csv', header=True)
    return df_merged_r

def get_next_batch(inputs, labels, batch_size, i ):
    # a helper function for the model where we can batch our data and labels appropriately
    batched_inputs = inputs[i*batch_size:i*batch_size + batch_size][:]
    batched_labels = labels[i*batch_size:i*batch_size + batch_size]
    return batched_inputs, batched_labels


