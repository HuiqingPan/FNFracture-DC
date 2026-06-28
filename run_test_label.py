from test_utils.tester_label import Tester_label
import warnings

warnings.filterwarnings('ignore')

tt = Tester_label()

tt.calc_label_metrics(
    path='',
    label_name='',
    pred_name='',
    print_type='ci',
    bootstrap=1000,
    frac=.8,
    round_digits=3
)
