import numpy as np
import openml
openml.config.server = "http://145.38.195.79/api/v1/xml"

datalist = openml.datasets.list_datasets(output_format="dataframe")

datalist['ValidInstNum'] = datalist['NumberOfInstances'] - datalist['NumberOfInstancesWithMissingValues']
datasets_list = datalist[(datalist['ValidInstNum'] > 100)
                             & (datalist['NumberOfNumericFeatures'] >= 3)]

datasets_list = datasets_list.sort_values('ValidInstNum')

for i, row in datasets_list.iterrows():
    ds = openml.datasets.get_dataset(row['did'])
    id = row['did']
    ds_name = ds.name
    target_name = ds.default_target_attribute
    try:
        dataset = ds.get_data()[0]
        target = dataset[target_name].astype(str)
        classes = np.unique(target)
        if len(classes) == 2:
            dataset.to_csv(f'datasets/binary_classification/{id}_{ds_name}.csv', index=False)
            print(f'{ds_name} - binary_classification')
        if 50 > len(classes) > 2:
            dataset.to_csv(f'datasets/multiclass_classification/{id}_{ds_name}.csv', index=False)
            print(f'{ds_name} - multiclass_classification')
        if len(classes) > 50:
            dataset.to_csv(f'datasets/regression/{id}_{ds_name}.csv', index=False)
            print(f'{ds_name} - regression')
    except KeyError as e:
        print(f'{ds_name} - skip - {e}')
        pass





