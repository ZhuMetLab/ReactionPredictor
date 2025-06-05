import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df = pd.read_csv('./ref_rp_data.csv')
    train_size = 0.8
    valid_size = 0.1
    test_size = 0.1
    temp_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, valid_df = train_test_split(temp_df, test_size=valid_size/(train_size+valid_size), random_state=42)
    df['set'] = 'train'
    df.loc[valid_df.index, 'set'] = 'valid'
    df.loc[test_df.index, 'set'] = 'test'
    df.to_csv('./rp_data_for_training.csv', index=False)

    print('ok!')
