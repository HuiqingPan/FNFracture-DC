from test_utils.compare import Compare
import warnings

warnings.filterwarnings('ignore')

label_name = 'true_garden'
Compare().compare(path1='',
                  path2='',
                  label_name=label_name,
                  proba_prefix='garden_prob_',
                  bootstrap=100, print_type='ci', frac=.8, side='right')
