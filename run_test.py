from test_utils.tester import Tester
from test_utils.create_baseline import Create_baseline
import warnings

warnings.filterwarnings('ignore')


tt = Tester()

tt.calc_all_metrics('',
                    label_name='true_garden', proba_prefix='garden_prob_',
                    print_type='ci', bootstrap=20, frac=.8)
