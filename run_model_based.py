import argparse
import pickle
import sys, os
import time
from recbole.config import Config
from recbole.quick_start import run_recbole
from recbole.data import data_preparation, create_dataset

if __name__ == '__main__':
    subset_list = [f"sample_{i}" for i in range(1, 200)]
    subset_folder_name = "URM_subsets_filtered"
    model_name = "FairGo_PMF"
    start_time = time.time()

    for subset_name in subset_list:

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', '-m', type=str, default=model_name, help='name of models')
        parser.add_argument('--dataset', '-d', type=str, default='ml-1M')
        parser.add_argument('--config_files', '-c', type=str, default='test.yaml')
        args = parser.parse_args()

        config_file_list = args.config_files.strip().split(' ') if args.config_files else None

        # Step 1: Split the dataset once using a sample model configuration
        sample_config = Config(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
        sample_config["data_path"] = 'dataset_v2/ml-1M'
        sample_config["data_path_inter"] = f'dataset_v2/ml-1M/{subset_folder_name}/{subset_name}.inter'
        dataset = create_dataset(sample_config)
        train_data, valid_data, test_data = data_preparation(sample_config, dataset)
        
        # Run the model using the pre-split data
        result = run_recbole(
            model=model_name, dataset=args.dataset, config_file_list=config_file_list,
            train_data=train_data, valid_data=valid_data, test_data=test_data
        )

        path = f"results/results_ml1m_URM_filtered_gender/result_{subset_name}_{model_name}.txt"
        with open(path, 'wb') as handle:
            pickle.dump(result, handle)

    print("Total Time: ", time.time() - start_time)
