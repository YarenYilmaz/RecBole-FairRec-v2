import os
import pickle
from recbole.test_results.visualization import plot_table_v2

def adjust_metrics(metrics, model_name, sens_attr):
    # Filter and rename metrics for FairGo model
    if 'FairGo' in model_name:
        metrics = {key.replace(" "+ sens_attr, "").replace("pretrain-", ""): value
                   for key, value in metrics.items() if "pretrain" in key}
    else:
        # Standardize other models' metrics by removing "gender" from names
        metrics = {key.replace(" "+sens_attr, ""): value for key, value in metrics.items()}
    return metrics


def read_txt(file_path, model_name, sens_attr):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            test_result = data['test_result'].get("sm-['"+sens_attr+"']", data['test_result'].get("none", data['test_result']))
            # Adjust metrics based on model specifics
            adjusted_result = adjust_metrics(test_result, model_name, sens_attr)
            return adjusted_result
        except pickle.UnpicklingError as e:
            print("Error unpickling file:", e)
            return {}


def plot_all_models(model_list, base_path):
    dicts = []
    # Get all subdirectories starting with 'results_'
    #subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("results_")]
    #subdirectories = ["results_BX_URM_filtered_age" , "results_ml1m_URM_filtered_age", "results_ml1m_URM_filtered_gender"]
    subdirectories = ["results_ml1m_URM_filtered_test"]
    for model_name in model_list:
        for sub_dir in subdirectories:
            for i in range(1, 55): # 61 adet sonuç için çalışabilir
                file_path = os.path.join(base_path, sub_dir, f"result_sample_{i}_{model_name}.txt")
                if os.path.exists(file_path):
                    dataset = sub_dir.split("_")[1] if len(sub_dir.split("_")) > 1 else "Unknown"
                    sensitive_feature = sub_dir.split("_")[-1].capitalize()
                    result_dict = read_txt(file_path, model_name, sensitive_feature.lower())
                    # Extract information from sub_dir to fill in the metadata


                    is_filtered = "Yes" if "filtered" in sub_dir else "No"
                    
                    result_dict.update({
                        "Model Name": model_name, 
                        "Subset ID": i, 
                        "Dataset": dataset,
                        "Sensitive Feature": sensitive_feature, 
                        "Is Filtered": is_filtered
                    })
                    dicts.append(result_dict)
                    print([key for key in result_dict.keys()])
    
    print(f"Metrics plotted")
    plot_table_v2(dicts)
model_list = ["FairGo_PMF"]
base_path = "./results"

plot_all_models(model_list, base_path)