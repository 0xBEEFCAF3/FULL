import datetime as dt
import pandas as pd


def as_date(ts):
    return(dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))


def clean_dataset(dataset_path):
    df = pd.read_json(dataset_path)
    cols = list(df.columns)

    # Drop meta data fields
    df.drop('txid', axis=1, inplace=True)
    df.drop('hash', axis=1, inplace=True)
    df.drop('version', axis=1, inplace=True)
    df.drop('locktime', axis=1, inplace=True)
    df.drop('vsize', axis=1, inplace=True)
    if 'conf' in cols:
        df.drop('conf', axis=1, inplace=True)
    # TODO do we really need to drop net difficulty
    df.drop(columns='networkdifficulty', inplace=True)
    # Sort by date value
    df = df.sort_values(by='mempooldate')
    # Datify mempool date
    df.mempooldate = pd.to_datetime(df.mempooldate.apply(as_date))

    df.set_index('mempooldate', drop=True, inplace=True)
    # group by each unique timestamp
    df = df.reset_index().groupby('mempooldate').mean()

    # resample to 15 sec intervals and foward fill when NA;s get created
    # pad -> forward fill
    # iloc -> first row is all na's skip that boi
    df = df.resample('15S').pad().iloc[1:, :]
    # Split data

    return df
