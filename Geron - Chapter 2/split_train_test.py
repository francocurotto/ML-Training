import numpy as np
import pandas as pd
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit

def split_train_test(data, test_ratio):
    #data = data.copy(deep=True) # avoids SettingWithCopyWarning
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def split_train_test_by_id(data, test_ratio, id_column):
    #data = data.copy(deep=True) # avoids SettingWithCopyWarning
    ids = data[id_column] 
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def stratify_income(data):
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0,1.5,3,4.5,6,np.inf],
                                labels=[1,2,3,4,5])
    return data

def split_train_test_strat(data, test_ratio, remove_cat):
    strat_data = stratify_income(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(strat_data, strat_data["income_cat"]):
        strat_train_set = strat_data.loc[train_index]
        strat_test_set = strat_data.loc[test_index]
    if remove_cat:
        for set_ in (strat_train_set, strat_test_set): 
            set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set

def get_strat_prop(data):
    return data["income_cat"].value_counts()/len(data)

