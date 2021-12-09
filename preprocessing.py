import pandas as pd
import functools


def get_data(rr_y1, rr_y2, rr_y3, rr_y4, tr_y1, tr_y2, tr_y3, tr_y4, rp, bool_tr2014):
    """
    The function takes in all of the input data and concatenates it into one dataframe. Then is reduces the dataframe,
    combining information into rows by team. It outputs a dataframe of [num_teams, num_data_points].
    :param rr_y1: All of the rr parameters are csv files of recruiting rankings across different years
    :param tr_y1: All of the rr parameters are csv files of team records across different years
    :param rp: A csv file with data for returning production of that year
    :param bool_tr2014: The csv file of team records from 2014 has an extra column to be dropped. This is a
    boolean that determines whether or not that csv file is present
    :return df_merged_r: A dataframe of [num_teams, num_data_points]
    """
    df_rr1 = pd.read_csv(rr_y1).drop(columns='year')
    df_rr2 = pd.read_csv(rr_y2).drop(columns='year')
    df_rr3 = pd.read_csv(rr_y3).drop(columns='year')
    df_rr4 = pd.read_csv(rr_y4).drop(columns='year')
    if bool_tr2014:
        df_tr1 = pd.read_csv(tr_y1).drop(columns='year')
    else:
        df_tr1 = pd.read_csv(tr_y1)
    df_tr2 = pd.read_csv(tr_y2)
    df_tr3 = pd.read_csv(tr_y3)
    df_tr4 = pd.read_csv(tr_y4)
    df_rp = pd.read_csv(rp).drop(columns='season')
    data_frames = [df_rr1, df_rr2, df_rr3, df_rr4, df_tr1, df_tr2, df_tr3, df_tr4, df_rp]
    df_merged_r = functools.reduce(lambda left, right: pd.merge(left, right, on=['team'], how='inner'), data_frames)
    return df_merged_r


def get_labels(exp_wins, tt, ppa):
    """
    This function concatenates csv files into a single dataframe, and then reduces the dataframe by team
    such that the dataframe is organized by rows where each row represents team data
    :param exp_wins: A csv file from the given year of expected wins per team
    :param tt: A csv file of team_talent rankings from either the 2018 or 2019 season
    :param ppa: A csv file of the predicted points added metrics from the 2018 or 2019 season
    :return df_merged_r: A dataframe of shape [num_teams, 10]
    """
    df_exp_wins_1 = pd.read_csv(exp_wins).drop(columns='year')
    df_tt_1 = pd.read_csv(tt).drop(columns='year')
    df_ppa_1 = pd.read_csv(ppa).drop(columns='conference')
    data_frames = [df_exp_wins_1, df_tt_1, df_ppa_1]
    df_merged_r = functools.reduce(lambda left, right: pd.merge(left, right, on=['team'], how='inner'), data_frames)
    return df_merged_r


def get_next_batch(inputs, labels, batch_size, i):
    """
    This function is a helper function so we can batch our inputs and labels for training and testing
    :param inputs: A numpy array of inputs shape of [num_teams, num_data_points]
    :param labels: A numpy array of labels shape of [num_teams]
    :param batch_size: An integer that is a model parameter
    :param i: an integer indicating the ith iteration of batching we are on
    :returns batched_inputs, batched_labels: Batched inputs is a numpy array of inputs of shape
    [batch_size, num_data_points]
    whereas batched_labels is a numpy array of shape [batch_size]
    """
    batched_inputs = inputs[i*batch_size:i*batch_size + batch_size][:]
    batched_labels = labels[i*batch_size:i*batch_size + batch_size]
    return batched_inputs, batched_labels
