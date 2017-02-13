"""
The utils module contains a collection of utility functions.
"""

def df_columns_required(df_name, df, col_type_dict):
    for k,v in col_type_dict.items():
        if not df_columns_exist(df, {k:v}):
            raise TypeError('Expected column "{0}" of type {1} in DataFrame "{2}."'.format(k, v, df_name))

def df_columns_exist(df, col_type_dict):
    for name, dtype in col_type_dict.items():
        if name not in df or df[name].dtype != dtype:
            return False
    return True

def df_nans_not_allowed(df_name, df):
    if df.isnull().values.any():
        raise TypeError('Found unexpected NaN values in DataFrame "{0}".'.format(df_name))

def df_columns_nans_not_allowed(df_name, df, col_list):
    for name in col_list:
        if df_has_nans(df, [name]):
            raise TypeError('Found unexpected NaN values in column "{0}" of DataFrame "{1}".'.format(name, df_name))

def df_columns_has_nans(df, col_list):
    for name in col_list:
        if df[name].isnull().values.any():
            return True
    return False

