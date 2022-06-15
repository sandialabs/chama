import unittest
import pandas as pd
import numpy as np
import chama.utils as cu

# sample_data = [{'Col1': 's1', 'Col2': 1.11, 'Col3', 1},
#               {'Col1': 's2', 'Col2': 2.22, 'Col3', 2},
#               {'Col1': 's3', 'Col2': 3.33, 'Col3', 3},
#               {'Col1': 's4', 'Col2': 4.44, 'Col3', 4}
#               ]

sample_data = {'Col1': ['s1', 's2', 's3'],
               'Col2': [1.11, 2.22, 3.33],
               'Col3': [1, 2, 3]}

sample_data2 = {'Col1': ['s1', 's2', 's3'],
                'Col2': [1.11, np.nan, 3.33],
                'Col3': [1, 2, 3]}


class UtilsTests(unittest.TestCase):
    
    def test_df_columns_required(self):
        df = pd.DataFrame.from_dict(sample_data)

        # test all good - raises exception (and fails test if not)
        cu._df_columns_required('sample_data', df, {'Col1': object,
                                                    'Col2': np.float64,
                                                    'Col3': np.int64})

        # test incorrect type - object
        with self.assertRaises(TypeError):
            cu._df_columns_required('sample_data', df, {'Col1': str,
                                                        'Col2': np.float64,
                                                        'Col3': np.int64})

        # test incorrect type - float
        with self.assertRaises(TypeError):
            cu._df_columns_required('sample_data', df, {'Col1': object,
                                                        'Col2': str,
                                                        'Col3': np.int64})

        # test incorrect type - int
        with self.assertRaises(TypeError):
            cu._df_columns_required('sample_data', df, {'Col1': object,
                                                        'Col2': np.float64,
                                                        'Col3': np.float64})

        # test passing multiple types
        cu._df_columns_required('sample_data', df, {'Col1': object,
                                                    'Col2': [np.int64,
                                                             np.float64],
                                                    'Col3': [np.int64,
                                                             np.float64]})

        # test passing multiple types with incorrect types
        with self.assertRaises(TypeError):
            cu._df_columns_required('sample_data', df, {'Col1': object,
                                                        'Col2': [np.int64,
                                                                 np.bool8],
                                                        'Col3': [np.int64,
                                                                 np.float64]})

    def test_df_columns_exist(self):
        df = pd.DataFrame.from_dict(sample_data)

        # test all good - raises exception (and fails test if not)
        self.assertTrue(cu._df_columns_exist(df, {'Col1': object,
                                                  'Col2': np.float64,
                                                  'Col3': np.int64}))

        # test incorrect type - object
        self.assertFalse(cu._df_columns_exist(df, {'Col1': str,
                                                   'Col2': np.float64,
                                                   'Col3': np.int64}))

        # test incorrect type - float
        self.assertFalse(cu._df_columns_exist(df, {'Col1': object,
                                                   'Col2': str,
                                                   'Col3': np.int64}))
        
        # test incorrect type - int
        self.assertFalse(cu._df_columns_exist(df, {'Col1': object,
                                                   'Col2': np.float64,
                                                   'Col3': np.float64}))

        # test passing multiple types
        self.assertTrue(cu._df_columns_exist(df, {'Col1': object,
                                                  'Col2': [np.int64,
                                                           np.float64],
                                                  'Col3': [np.int64,
                                                           np.float64]}))

        # test passing multiple types with incorrect types
        self.assertFalse(cu._df_columns_exist(df, {'Col1': object,
                                                   'Col2': [np.int64,
                                                            np.bool8],
                                                   'Col3': [np.int64,
                                                            np.float64]}))

    def test_df_columns_has_nans(self):
        df = pd.DataFrame.from_dict(sample_data2)

        # test for NAN in col2 (True)
        self.assertTrue(cu._df_columns_has_nans(df, 'Col2'))

        # test for NAN in col3 (False)
        self.assertFalse(cu._df_columns_has_nans(df, 'Col3'))

        # test for NAN in col2 or col3 (True)
        self.assertTrue(cu._df_columns_has_nans(df, ['Col2', 'Col3']))

        # test for NAN in col1 or col3 (False)
        self.assertFalse(cu._df_columns_has_nans(df, ['Col1', 'Col3']))
        
        # test for exception if column name does not exist
        with self.assertRaises(TypeError):
            cu._df_columns_has_nans(df, ['Col1', 'ColDoesNotExist'])
        with self.assertRaises(TypeError):
            cu._df_columns_has_nans(df, 'ColDoesNotExist')

    def test_df_columns_nans_not_allowed(self):
        df = pd.DataFrame.from_dict(sample_data2)

        # test for NAN in col2 (True)
        with self.assertRaises(TypeError):
            cu._df_columns_nans_not_allowed('sample_data2', df, 'Col2')

        # test for NAN in col3 (False)
        cu._df_columns_nans_not_allowed('sample_data2', df, 'Col3')

        # test for NAN in col2 or col3 (True)
        with self.assertRaises(TypeError):
            cu._df_columns_nans_not_allowed('sample_data2', df,
                                            ['Col2', 'Col3'])

        # test for NAN in col1 or col3 (False)
        cu._df_columns_nans_not_allowed('sample_data2', df, ['Col1', 'Col3'])
        
        # test for exception if column name does not exist
        with self.assertRaises(TypeError):
            cu._df_columns_nans_not_allowed('sample_data2', df,
                                            ['Col1', 'ColDoesNotExist'])
        with self.assertRaises(TypeError):
            cu._df_columns_nans_not_allowed('sample_data2', df,
                                            'ColDoesNotExist')

    def test_df_nans_not_allowed(self):
        df = pd.DataFrame.from_dict(sample_data)
        df2 = pd.DataFrame.from_dict(sample_data2)

        cu._df_nans_not_allowed('sample_data', df)

        with self.assertRaises(TypeError):
            cu._df_nans_not_allowed('sample_data2', df2)

if __name__ == "__main__":
    unittest.main()
