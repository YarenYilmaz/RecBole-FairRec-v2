# @Time   : 2020/7/20
# @Author : sahanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse

from recbole.quick_start import run_recbole
import pickle

if __name__ == '__main__':
    import sys, os

    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FairGo_PMF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1M', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default='test.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    returned_result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

    # Save to a file
    with open('results/output.pkl', 'wb') as file:
        pickle.dump(returned_result, file)

    print("Data saved to output.pkl")


