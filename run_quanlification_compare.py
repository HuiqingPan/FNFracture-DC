from test_utils.quanlification_compare import Compare
import warnings

warnings.filterwarnings('ignore')

Compare().compare(path1='',
                  path2='',
                  label_name1='pred_angle', label_name2='pred_angle',
                  true_angle='true_angle',
                  bootstrap=100, print_type='ci', frac=.8, side='right')