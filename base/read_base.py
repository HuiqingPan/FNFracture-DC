import os

import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.utils import resample

class Read_base:
    def fill(self, df: pd.DataFrame, col_name: list, fill_type: str = 'median') -> pd.DataFrame:
        """
        Fill missing values in the specified columns of the DataFrame.

        Parameters:
        df (pd.DataFrame): The input dataframe with missing values.
        col_name (list): List of column names where missing values should be filled.
        fill_type (str): The method used to fill missing values. Options are 'median' or 'mice'.
                         'median' fills missing values with the median of the column.
                         'mice' uses Multiple Imputation by Chained Equations for filling missing values.

        Returns:
        pd.DataFrame: The dataframe with missing values filled.
        """

        if fill_type == 'median':
            # Median filling: Fill missing values with the median of each specified column
            for col in col_name:
                if col in df.columns:
                    # Count missing values before filling
                    missing_before = df[col].isna().sum()

                    # Fill missing values with the median
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

                    # Count missing values after filling
                    missing_after = df[col].isna().sum()

                    # Number of filled data points
                    filled_count = missing_before - missing_after
                    print(f"Column '{col}' filled {filled_count} missing data points.")

                else:
                    print(f"Warning: Column '{col}' does not exist in the DataFrame.")

        elif fill_type == 'mice':
            # MICE (Multiple Imputation by Chained Equations): Uses an iterative method to predict and fill missing values
            for col in col_name:
                if col in df.columns:
                    # Count missing values before filling
                    missing_before = df[col].isna().sum()

                    # Perform MICE imputation
                    mice_imputer = IterativeImputer()
                    df[col] = mice_imputer.fit_transform(df[[col]])

                    # Count missing values after filling
                    missing_after = df[col].isna().sum()

                    # Number of filled data points
                    filled_count = missing_before - missing_after
                    print(f"Column '{col}' filled {filled_count} missing data points.")

                else:
                    print(f"Warning: Column '{col}' does not exist in the DataFrame.")

        else:
            print(f"Error: Unsupported fill_type '{fill_type}'. Please choose 'median' or 'mice'.")

        return df

    def outlier(self, col: pd.Series, sd: float):
        """
        Replace outliers with NaN. An outlier is defined as a value that is greater than
        the given number of standard deviations away from the mean.

        Parameters:
        col (pd.Series): The column of data where outliers should be removed.
        sd (float): The standard deviation threshold. Values beyond mean ± (sd * standard deviation)
                    will be considered outliers.

        Returns:
        pd.Series: The column with outliers replaced by NaN.
        """

        # Calculate the mean and standard deviation of the column
        mean = col.mean()
        std_dev = col.std()

        # Identify outliers: values outside mean ± (sd * standard deviation)
        lower_bound = mean - sd * std_dev
        upper_bound = mean + sd * std_dev
        mask = (col < lower_bound) | (col > upper_bound)
        # Replace outliers with NaN
        new_col = col.copy()
        new_col[mask] = np.nan
        drop_number = [i for i in mask if i]
        return new_col, len(drop_number)

    def concat_str(self, obj: list or pd.DataFrame, col_name: list = None,
                   special_symbol: str = '<SEP>') -> str or list:#这种拼接对数据预处理有什么用吗
        """
        Concatenate strings from specific columns of a DataFrame or a list of strings.

        Parameters:
        obj (list or pd.DataFrame): Input can be a DataFrame or a list of strings.
                                     If it's a DataFrame, col_name should be specified.
        col_name (list): List of column names in the DataFrame to be concatenated.
                         Ignored if obj is a list.
        special_symbol (str): Symbol used to concatenate the strings. Default is '<SEP>'.

        Returns:
        str or pd.Series: If obj is a list, return a single concatenated string.
                          If obj is a DataFrame, return a Series where each entry is the concatenated string for that row.
        """

        if isinstance(obj, pd.DataFrame):
            # If obj is a DataFrame, concatenate values from the specified columns
            result = []
            for _, row in obj.iterrows():
                # Filter out None or NaN values from the row for the specified columns
                values = [str(row[col]) for col in col_name if pd.notna(row[col])]
                # Concatenate values using special_symbol
                concatenated_str = special_symbol.join(values)
                result.append(concatenated_str)
            return result  # Return as a list of concatenated strings

        elif isinstance(obj, list):
            # If obj is a list, concatenate all elements using special_symbol
            return special_symbol.join([str(item) for item in obj])

        else:
            # If neither list nor DataFrame is passed, raise an error
            raise ValueError("Input must be a list or DataFrame.")

    def find_fig_path(self, ID: str, dirs: list, fig_type: str = 'png') -> str:
        """
        Find the path to the image file with the given ID in the provided directories.
        Supports multiple image formats.

        Parameters:
        ID (str): The ID of the image to search for.
        dirs (list): List of directories to search for the image file.
        fig_type (str): The image format (e.g., 'png', 'jpg', 'jpeg'). Default is 'png'.

        Returns:
        str: The full path to the image file if found, otherwise an empty string.
        """

        # Possible image extensions
        possible_extensions = [fig_type, 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']

        # Loop through each directory in dirs
        for dir_path in dirs:
            # Check if the directory exists
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist.")
                continue

            # Check each possible extension
            for ext in possible_extensions:
                file_name = f"{ID}.{ext}"
                file_path = os.path.join(dir_path, file_name)

                # If the file exists, return its path
                if os.path.isfile(file_path):
                    return file_path

        # Return an empty string if no file is found
        return ""

    def check_feature_dimension(self, features, expected_length):
        for key, value in features.items():
            if value is None:
                continue
            print(len(value))
            actual_length = len(value) if isinstance(value, (list, np.ndarray, pd.Series)) else value.shape[0]
            if actual_length == expected_length:
                print('=====================Feature dimension correct.=====================')
            elif actual_length != expected_length:
                raise ValueError(
                    f'Feature {key} has {actual_length} dimension, not consistent with {expected_length}'
                )

    def oversample(self, features: dict) -> dict:
        """
        Perform oversampling on the provided features dictionary.

        Parameters:
        features (dict): Dictionary containing various features including 'labels'.

        Returns:
        dict: Dictionary with oversampled features.
        """
        # Ensure labels are present
        labels = features.get('labels')
        if labels is None:
            raise ValueError("Labels are required for oversampling")

        # Check if it's a multi-label task
        is_multilabel = len(labels.shape) > 1
        print(is_multilabel)

        # Oversampling logic for multi-label and single-label cases
        if is_multilabel:
            oversampled_features = self._oversample_multilabel(features)
        else:
            oversampled_features = self._oversample_single_label(features, labels)

        return dict(oversampled_features)

    def _oversample_multilabel(self, features: dict) -> dict:
        """
        Oversample features for multi-label tasks based on joint label distribution.

        Parameters:
        features (dict): Dictionary of features including multi-label 'labels'.

        Returns:
        dict: Oversampled features.
        """
        labels = features['labels']
        unique_labels, label_counts = np.unique(labels, axis=0, return_counts=True)
        majority_class_count = max(label_counts)

        # Collect oversampled indices and oversampled labels
        oversampled_indices = []
        for label in unique_labels:
            label_indices = np.where(np.all(labels == label, axis=1))[0]
            oversampled_indices.extend(resample(label_indices, replace=True,
                                                n_samples=majority_class_count, random_state=42))

        return self._replicate_features(features, oversampled_indices)

    def _oversample_single_label(self, features: dict, labels: np.ndarray) -> dict:
        """
        Oversample features for single-label tasks based on label distribution.

        Parameters:
        features (dict): Dictionary of features including single-label 'labels'.
        labels (np.ndarray): Single-label array.

        Returns:
        dict: Oversampled features.
        """
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        majority_class_count = max(label_counts)

        # Collect oversampled indices
        oversampled_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            oversampled_indices.extend(resample(label_indices, replace=True,
                                                n_samples=majority_class_count, random_state=42))

        return self._replicate_features(features, oversampled_indices)

    def _replicate_features(self, features: dict, indices: list) -> dict:
        """
        Replicate features based on provided indices.

        Parameters:
        features (dict): Dictionary of features to replicate.
        indices (list): Indices to replicate.

        Returns:
        dict: Replicated features.
        """
        oversampled_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                oversampled_features[key] = value[indices]
            elif isinstance(value, list):
                oversampled_features[key] = [value[i] for i in indices]
            else:
                oversampled_features[key] = value  # If not array/list, leave as is
        return oversampled_features

    # def check_feature_dimension(self, features: dict, suppose_length: int):
    #     #和上方oversample同样的问题，是需要自己把Dataframe转换成字典格式吗？字典的键是每一个Dataframe的列名，字典的值是原Dataframe里每行对应到这一列的值吗？再把值转成ndarray？
    #     # Validate that the number of rows in non-None features matches the dataframe length
    #     for key, value in features.items():
    #         if value is not None:
    #             if isinstance(value, np.ndarray):
    #                 if value.shape[0] != suppose_length:
    #                     raise ValueError(
    #                         f"Length mismatch: The number of rows in '{key}' does not match the dataframe.")
    #             elif isinstance(value, list):
    #                 if len(value) != suppose_length:
    #                     raise ValueError(
    #                         f"Length mismatch: The number of items in '{key}' does not match the dataframe.")
