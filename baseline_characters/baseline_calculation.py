import pandas as pd
from tableone import TableOne
import numpy as np
from scipy import stats

class create_tableone:

    def create_tableone(self, file_path, total_columns: list, categorical: list, continuous: list, group=False, group_column=None,
                        nonnormal=None,
                        original_name=None, current_name=None, pval=True,
                        save_path1 = None,
                        save_path2 = None):
        """
        创建TableOne数据统计报告。
    
        参数:
        - df: 输入的DataFrame数据集
        - columns: 需要统计的列名
        - categorical: 需要进行分类统计的列名
        - pval: 是否显示p值，默认为True
    
        返回:
        - tableone: TableOne对象，包含统计结果
        """
        df = pd.read_excel(file_path)
    
        if group:
            group_column = group_column
        nonnormal = nonnormal
        rename = None
        if original_name is not None and current_name is not None:
            rename = {original_name: current_name}# rename={'death': 'mortality'}
    
        table_baseline = TableOne(df, columns=total_columns, categorical=categorical, continuous=continuous, groupby=group_column,
                                  nonnormal=nonnormal, rename=rename, pval=pval)
    
        table_baseline.to_excel(save_path1)
    
        print(table_baseline.tabulate(tablefmt="fancy_grid"))
    
        effect_df = None
        if group:
            effect_df = self.calculate_effect_sizes(
                df=df,
                group_column=group_column,
                categorical=categorical,
                continuous=continuous
            )
    
            print("\n===== Effect Size Results =====\n")
            print(effect_df.to_string(index=False))
            effect_df.to_excel(save_path2)
    
        return table_baseline, effect_df
    
    def calculate_effect_sizes(self, df, group_column, categorical, continuous):
        """
        计算效应量：
        - 连续变量：η² from one-way ANOVA
        - 分类变量：Cramér's V from chi-square test
        """
        results = []
    
        # 去掉 group_column 本身，避免重复分析
        continuous_vars = [col for col in continuous if col != group_column]
        categorical_vars = [col for col in categorical if col != group_column]
    
        # 连续变量：ANOVA + eta squared
        for var in continuous_vars:
            temp = df[[group_column, var]].dropna()
    
            # 至少要有2组
            groups = [g[var].values for _, g in temp.groupby(group_column)]
            if len(groups) < 2:
                continue
    
            try:
                f_stat, p_val = stats.f_oneway(*groups)
    
                # eta squared
                grand_mean = temp[var].mean()
                ss_between = sum(
                    len(g[var]) * (g[var].mean() - grand_mean) ** 2
                    for _, g in temp.groupby(group_column)
                )
                ss_total = sum((temp[var] - grand_mean) ** 2)
    
                eta_sq = ss_between / ss_total if ss_total != 0 else np.nan
    
                results.append({
                    "variable": var,
                    "type": "continuous",
                    "test": "one-way ANOVA",
                    "p_value": p_val,
                    "effect_size": eta_sq,
                    "effect_name": "eta_squared",
                    "magnitude": self.interpret_eta_squared(eta_sq)
                })
            except Exception as e:
                results.append({
                    "variable": var,
                    "type": "continuous",
                    "test": "one-way ANOVA",
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "effect_name": "eta_squared",
                    "magnitude": f"error: {e}"
                })
    
        # 分类变量：chi-square + Cramér's V
        for var in categorical_vars:
            temp = df[[group_column, var]].dropna()
    
            if temp.empty:
                continue
    
            contingency_table = pd.crosstab(temp[group_column], temp[var])
    
            # 至少要形成 2x2 或更大表
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue
    
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    
                n = contingency_table.to_numpy().sum()
                r, k = contingency_table.shape
    
                cramer_v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if n > 0 else np.nan
    
                results.append({
                    "variable": var,
                    "type": "categorical",
                    "test": "chi-square",
                    "p_value": p_val,
                    "effect_size": cramer_v,
                    "effect_name": "cramers_v",
                    "magnitude": self.interpret_cramers_v(cramer_v)
                })
            except Exception as e:
                results.append({
                    "variable": var,
                    "type": "categorical",
                    "test": "chi-square",
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "effect_name": "cramers_v",
                    "magnitude": f"error: {e}"
                })
    
        effect_df = pd.DataFrame(results)
        return effect_df
    
    @staticmethod
    def interpret_eta_squared(value):
        if pd.isna(value):
            return "NA"
        elif value < 0.01:
            return "negligible"
        elif value < 0.06:
            return "small"
        elif value < 0.14:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def interpret_cramers_v(value):
        if pd.isna(value):
            return "NA"
        elif value < 0.1:
            return "negligible"
        elif value < 0.3:
            return "small"
        elif value < 0.5:
            return "medium"
        else:
            return "large"

