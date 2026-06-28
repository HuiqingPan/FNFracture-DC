import pandas as pd
from statsmodels.tools import categorical
from tableone import TableOne
from baseline_characters.baseline_calculation import create_tableone
from reader.reader import Reader
from arguments.config_baseline import Config_read
import warnings

warnings.filterwarnings('ignore')

read_table = Config_read()
file_path = 'data/baseline.xls'

total_columns = read_table.total_columns
categorical_varaibles = read_table.categorical_columns
continuous_varaibles = read_table.continuous_columns

baseline = create_tableone()
baseline.create_tableone(file_path, total_columns=total_columns, categorical=categorical_varaibles, continuous=continuous_varaibles,
                         group=True,
                         group_column='training_set',
                         pval=True,
                         save_path1='results/baseline/baseline.xlsx',
                         save_path2='results/baseline/effect_size.xlsx')


