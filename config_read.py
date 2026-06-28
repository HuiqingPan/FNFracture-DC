from 分型与坏死预测.base.base import Base


class Config_read(Base):
    def __init__(self):
        self.feature_dict = self.load_dict('feature_dict.xlsx')
        self.structured_name = self.get_structured_names(self.feature_dict, type_name=['s_lasso'])  # 所有结构化数据的列名，手动输入type_name则指定类型
        self.text_name = self.feature_dict['text']
        self.fig_dirs = self.feature_dict['fig']
        self.label_name = ['fracture', 'Garden', 'Pauwels']

data = Config_read()