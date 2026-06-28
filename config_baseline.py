from 分型与坏死预测.base.base import Base


class Config_read(Base):
    def __init__(self):
        self.feature_dict = self.load_dict('baseline_dict.xlsx')
        self.total_columns = self.get_structured_names(self.feature_dict, type_name=['colocolumns'])  # 所有结构化数据的列名，手动输入type_name则指定类型
        self.categorical_columns = self.get_structured_names(self.feature_dict, type_name=['categorical'])
        self.continuous_columns = self.get_structured_names(self.feature_dict, type_name=['continuous'])

data = Config_read()