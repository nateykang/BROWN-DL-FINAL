#preprocessing
import pandas as pd

def get_data(rr_y1,rr_y2,rr_y3,rr_y4,tr_y1,tr_y2,tr_y3,tr_y4,rp):
    # get_data concatenates all the input data into one dataframe
    # dataframe will consist of teams in every team record dataset
    # rr_y# is the recruiting ranking of the team # years from the test set (eg for testing on 2018, y1 is 2014)
    # tr_y# is the team record # years from the test set (eg for testing on 2018, y1 is 2014)
    # rp is returning production of the team in the test year
    df_rr1 = pd.read_csv(rr_y1)
    df_rr2 = pd.read_csv(rr_y2)
    df_rr3 = pd.read_csv(rr_y3)
    df_rr4 = pd.read_csv(rr_y4)
    df_tr1 = pd.read_csv(tr_y1)
    df_tr2 = pd.read_csv(tr_y2)
    df_tr3 = pd.read_csv(tr_y3)
    df_tr4 = pd.read_csv(tr_y4)
    df_rp = pd.read_csv(rp)
    data_frames = [df_rr1,df_rr2,df_rr3,df_rr4,df_tr1,df_tr2,df_tr3,df_tr4,df_rp]
    df_merged = pd.reduce(lambda  left,right: pd.merge(left,right,on=['team'],how='outer'), data_frames)
    print(df_merged.head)
    pass
