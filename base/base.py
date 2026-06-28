import joblib
import numpy as np
import pandas as pd
from pandas.tests.groupby.conftest import dropna

from base.wash_base import Wash_base
from base.merge_base import Merge_base
from base.read_base import Read_base
from base.test_base import Test_base
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from PIL import Image
import os.path


class Base(Wash_base, Merge_base, Read_base, Test_base):
    def load_df(self, path: str or pd.DataFrame, encoding='utf-8', verbose=False):
        """
        Loads a DataFrame from a file (CSV, Excel) or directly from an existing DataFrame.

        This function supports loading CSV, Excel (including .xlsx, .xls, .xlsm, .xlsb),
        or directly using an existing DataFrame.
    
        Parameters:
        path (str or pd.DataFrame): File path or DataFrame.
        encoding (str): The encoding to use for CSV files (default is 'utf-8').
    
        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        if isinstance(path, (pd.DataFrame, pd.Series)):
            if verbose:
                print(f'Getting an existed Dataframe, with {len(path)} data')
            return path
    
        else:
            # Handle CSV files
            if path.endswith('.csv'):
                df = pd.read_csv(path, encoding=encoding, dtype={'ID': str})
                if verbose:
                    print(f'Loading a Dataframe, with {len(df)} data')
                return df


            # Handle Excel files (xls, xlsx, xlsm, xlsb)
            elif path.endswith(("xls", "xlsx", "xlsm", "xlsb")):
                df = pd.read_excel(path, dtype={'ID': str})
                if verbose:
                    print(f'Loading a Dataframe, with {len(df)} data')
                return df
    
            else:
                raise ValueError(
                    f"Unsupported file type: {path.split('.')[-1]}. "
                    f"Only 'csv', 'xls', 'xlsx', 'xlsm', and 'xlsb' formats are supported.")
    
    def save_df(self, df: pd.DataFrame, save_path: str, encoding='utf-8'):
        """
        Saves the DataFrame to a file in the specified format (CSV, Excel, or TXT).
    
        Parameters:
        df (pd.DataFrame): The DataFrame to save.
        save_path (str): The path to save the DataFrame.
        encoding (str): The encoding to use for CSV files (default is 'utf-8').
        """
        if save_path is not None:
            if save_path.endswith('.csv'):
                df.to_csv(save_path, encoding=encoding, index=False)
            elif save_path.endswith(('.xlsx', '.xls')):
                df.to_excel(save_path, index=False)
            elif save_path.endswith('.txt'):
                df.to_csv(save_path, encoding=encoding, sep='\t')
            else:
                raise ValueError("Unsupported file format.")
            print(f"File saved at {save_path}. Total rows: {len(df)}")
    
    def load_dict(self, path: str) -> dict:
        """
        Loads a dictionary from a file (CSV or Excel) where each column name maps to a list of its non-null values.
    
        Parameters:
        path (str): The file path for the dictionary.
    
        Returns:
        dict: A dictionary mapping each unique column name to its non-null values.
        """
        df = self.load_df(path)
        #没必要先转成集合再转成列表？下面写的岂不更方便？还不会打乱顺序
        # return {col: df[col].dropna().nunique().tolist() for col in df}
        return {col: list(set(df[col].dropna().tolist())) for col in df}
    
    def _read_structured(self, path, col_name: list, fill_type: str = None, sd: float = None,
                         save_path=None) -> np.ndarray:
        """
        Reads a structured dataset from the specified file path, applies outlier detection and imputation,
        and returns the data as a numpy array.
    
        Parameters:
        path (str): The file path of the dataset to be loaded.
        col_name (list): A list of column names to process from the dataframe.
        fill_type (str, optional): The method used to fill missing values (e.g., 'median', 'mean').
        sd (float, optional): The number of standard deviations for outlier detection. If None, no outlier handling is applied.
        save_path (str, optional): The path to save the processed dataframe.
    
        Returns:
        np.ndarray: A numpy array containing the processed data from the specified columns.
        """
    
        # Load the dataframe from the given file path
        df = self.load_df(path)
    
        # Handle outliers by replacing values that are outside the specified standard deviation range
        if sd is not None:
            drop_outlier = {}
            # Process each column specified in col_name
            for c in col_name:
                if c in df.columns:  # Check if the column exists in the dataframe
                    df[c], drop_number = self.outlier(df[c], sd=sd)
                    drop_outlier[c] = drop_number
    
            for k, v in drop_outlier.items():
                print(f'Dropping {v} outliers for {k}')
        else:
            print('Warning: Not doing outlier handling!!')
    
        eligble_name = [c for c in col_name if c in df.columns]
        # Handle missing values by applying the fill method if specified
        if fill_type is not None:
            df = self.fill(df, col_name=col_name, fill_type=fill_type)
        else:
            print('Warning: Not doing missing value filling!!')
    
        # Convert the selected columns to a numpy array
        sturctured = df[col_name].round(2).to_numpy()
    
        # Save the processed dataframe to the specified path (if provided)
        if save_path:
            self.save_df(df, save_path)
    
        print(f'Loading structured features from {", ".join([e for e in eligble_name])}, '
              f'with a total {len(eligble_name)} features')
    
        if not len(eligble_name) == len(col_name):
            print(f'Warning: These features are not compatible '
                  f'{set(col_name) - set(eligble_name)}')
        else:
            print('=====================All structured features successfully loaded.=====================')
    
        return sturctured
    
    def _read_text(self, path, col_name: list, special_symbol: str = '<SEP>') -> list:
        """
        Reads a dataset, concatenates specified text columns, and returns a list of concatenated strings.
    
        Parameters:
        path (str): The file path of the dataset to be loaded.
        col_name (list): A list of column names containing text data to concatenate.
        special_symbol (str): The symbol used to concatenate the text columns (default is '<SEP>').
    
        Returns:
        list: A list of concatenated strings for each row in the dataset.
        """
    
        # Load the dataframe from the given file path
        df = self.load_df(path)
    
        # Concatenate the specified text columns using the given separator
        texts = self.concat_str(df, col_name=col_name, special_symbol=special_symbol)#这有什么用呢？疑问同read_base中的
        print(f'Loading texts from {", ".join(col_name)}')
        return texts
    
    def _read_fig_path(self, path, ID, dirs, fig_type: str = 'png') -> list:
        """
        Reads a dataset and retrieves the file paths of images based on the given ID column.
    
        Parameters:
        path (str): The file path of the dataset to be loaded.
        ID (str): The column name containing the IDs of the images.
        dirs (list): A list of directories to search for the image files.
        fig_type (str): The type of image file (default is 'png').
    
        Returns:
        list: A list of image file paths corresponding to the IDs in the dataset.
        """
    
        # Load the dataframe from the given file path
        df = self.load_df(path)
    
        # Extract the list of IDs from the dataframe
        # ID_list = list(df[ID].astype(str))
        ID_list = [str(id).zfill(3) for id in df[ID]]
        print(ID_list)
    
        # Initialize an empty list to store the figure paths
        fig_path_list = []
    
        # For each ID, find the corresponding image file path
        for id in ID_list:
            fig_path_list.append(self.find_fig_path(ID=id, dirs=dirs, fig_type=fig_type))
    
        # Ensure the number of found image paths matches the number of rows in the dataframe
        assert len(df) == len(fig_path_list)
    
        return fig_path_list
    
    def _read_labels(self, path, col_name) -> list:
        """
        Reads a dataset and returns the labels from a specified column.
    
        Parameters:
        path (str): The file path of the dataset to be loaded.
        col_name (str): The column name containing the labels.
    
        Returns:
        list: A list of labels from the specified column in the dataset.
        """
    
        # Load the dataframe from the given file path
        df = self.load_df(path)
    
        # Return the values of the specified column as a numpy array
        return df[col_name].to_numpy()
    
    def get_structured_names(self, feature_dict: dict, type_name: list = None, prefix='s_') -> list:
        """
        Retrieves all the structured feature names from the provided feature dictionary
        that start with the specified prefix.
    
        Parameters:
        feature_dict (dict): A dictionary containing features (e.g., keys are feature names).
        prefix (str): The prefix that each structured feature name should start with (default is 's_').
    
        Returns:
        list: A list of all feature names that start with the given prefix.
        """
        # Initialize an empty list to store the feature names that match the prefix
        if isinstance(feature_dict, str):
            feature_dict = self.load_dict(feature_dict)
        all_structured = []
    
        if type_name is None:
            # Iterate over keys in feature_dict that start with the specified prefix
            for k in [k for k in feature_dict if k.startswith(prefix)]:
                # For each matching key, extend the list with its associated values (which are assumed to be lists)
                all_structured.extend(
                    feature_dict[k])  # Fix: changed from list[feature_dict[k]] to list(feature_dict[k])
        else:
            for k in type_name:
                all_structured.extend(feature_dict[k])
    
        # Return the list of all structured feature names
        return all_structured
    
    def _split(self, path, test_size, label_col=None, train_path='train.csv', test_path='test.csv', seed=18):
        """
        Function:
        1. Reads data from an Excel file based on the input parameters.
        2. If a label column is provided, performs stratified splitting based on the label; otherwise, performs random splitting.
        3. Saves the training and testing datasets to the specified paths.
    
        Input parameters:
        - path: Path to the Excel file containing the data to be processed.
        - test_size: Proportion of the dataset to be used as the test set (float), range: 0 < test_size < 1.
        - label_col: List of label columns to be used for stratified splitting (default is None). If None, random splitting will be used.
        - train_path: Path where the training dataset will be saved (default is 'train.csv').
        - test_path: Path where the test dataset will be saved (default is 'test.csv').
        - seed: Random seed for reproducibility (default is 18).
    
        Output:
        - Saves the training and testing datasets to the specified paths.
        """
    
        # 1. Read the Excel file
        try:
            df = self.load_df(path)
            print(f"Successfully loaded the dataset: {path}")
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            return
    
        # 2. Check if the dataset is empty
        if df.empty:
            print("The dataset is empty. Please check the input Excel file.")
            return
    
        # 3. Perform stratified or random splitting based on the label column
        if label_col is not None:
            print(f"Performing stratified splitting based on label column(s): {label_col}...")
            # Ensure that the label column exists in the dataset
            if not all(label in df.columns for label in label_col):
                print(f"Error: The label column(s) {label_col} do not exist in the dataset.")
                return
    
            # Perform stratified sampling using train_test_split
            train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])
            print("Stratified splitting based on label(s) succeeded.")
        else:
            print(f"Performing random splitting...")
            # If no label column is provided, perform simple random splitting
            train, test = train_test_split(df, test_size=test_size, random_state=seed)
            print("Random splitting succeeded.")
    
        # 5. Print the basic information about the training and test sets
        print(f"Training set size: {train.shape}")
        print(f"Test set size: {test.shape}")
    
        assert train.shape[1] == df.shape[1]
        assert test.shape[1] == df.shape[1]
        print('=====================Split is successful.=====================')
    
        # 6. Save the training and test sets to the specified paths
        try:
            self.save_df(train, train_path)
            self.save_df(test, test_path)
        except Exception as e:
            print(f"Error occurred while saving the datasets: {e}")
    
        return train, test
    
    def load_ml(self, model_path: str):
        """
        Loads the model from the specified path.
    
        Parameters:
        - model_path (str): Path to the saved model file.
        """
        print(f"Loading model from {model_path}...")
        return joblib.load(model_path)


    def save_ml(self, model: BaseEstimator, model_path: str):
        joblib.dump(model, model_path)
        print(f'Saving model to {model_path}')
    
    def _print_mode(self, array: np.ndarray, mode: str = 'ci', round: int = 3, name: str = ''):
        means = np.mean(array)
        if mode == 'ci':
            lower, upper = np.percentile(array, [2.5, 97.5])
            print(f'{name}: {means:.{round}f} ({lower:.{round}f}–{upper:.{round}f})')
        elif mode == 'sd':
            std = np.std(array)
            print(f"{name}: {means:.{round}f} ({std:.{round}f})")
    
    def read_rd(self, path, proba_prefix='proba_', label_name='labels'):
        """
        读取存储的结果数据，并返回预测概率和真实标签。
        :param path: 存储结果的路径
        :param proba_prefix: 预测概率列的前缀
        :param label_name: 标签列的名称
        :return: probas, labels
        """
        # 读取CSV文件
        results_df = self.load_df(path)
        results_df = results_df.dropna(subset=[label_name])
        probas = results_df.filter(regex=proba_prefix).to_numpy()
        labels = results_df[label_name].to_numpy()
        return probas, labels
    
    def _convert_to_tensor_format(self, labels: np.array, fig_path_list: list):
    
        # labels_indices = np.argmax(labels, axis=1)
        labels = torch.LongTensor(labels)
    
        transform = transforms.Compose([transforms.ToTensor()])
    
        image_tensors = []
        valid_count = 0
    
        for image_path in fig_path_list:
            try:
                if not os.path.exists(image_path):
                    print(f'图形不存在：{image_path}')
                    continue
    
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
                image_tensors.append(image_tensor)
    
                valid_count += 1
    
            except:
                print(f'无法读取图像：{image_path}')
                continue
            # print(self.image_tensors)
    
        print(f'成功转换{valid_count}张图像')
    
        return labels, image_tensors
