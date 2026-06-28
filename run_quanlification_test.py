import warnings

from 分型与坏死预测.test_utils.quanlification_tester import Tester

warnings.filterwarnings('ignore')

tt = Tester()

tt.calc_all_metrics('',
                    sheet_name='',
                    y_true='true_angle',
                    y_pred='pred_angle',
                    bootstrap=20, frac=.8)


