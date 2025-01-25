import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

openml.config.server = "http://145.38.195.79/api/v1/xml"


datalist = openml.datasets.list_datasets(output_format="dataframe")
datalist['ValidInstNum'] = datalist['NumberOfInstances'] - datalist['NumberOfInstancesWithMissingValues']
datasets_list = datalist[(datalist['NumberOfClasses'] == 0)
                         & (datalist['ValidInstNum'] < 60000)
                         & (datalist['ValidInstNum'] > 100)
                         & (datalist['NumberOfNumericFeatures'] >= 3)]
datasets_list = datasets_list.sort_values('NumberOfInstances')
#datasets_list.to_csv('datasets_list(60k).csv')

for i, ds in datasets_list.iterrows():
    dataset = openml.datasets.get_dataset(ds['did'])
    dataset_name = dataset.name
    target_name = dataset.default_target_attribute
    dataset_df = dataset.get_data()[0]
    dataset_df = dataset_df.dropna()
    for column in dataset_df.columns:
        if column != target_name:
            if dataset_df[column].dtype.name in ['object', 'category']:
                try:
                    dataset_df[column] = dataset_df[column].astype(int)
                except Exception as e:
                    try:
                        encoder = LabelEncoder()
                        encoder.fit_transform(dataset_df[column].to_frame())
                        dataset_df[column] = encoder.transform(dataset_df[column].to_frame())
                    except Exception as e:
                        pass
    dataset_df = dataset_df.apply(pd.to_numeric, errors='coerce')
    dataset_df = dataset_df.dropna()

