import argparse
import json
import os
import pickle
import time
from pathlib import Path
import subprocess
import torch.multiprocessing as mp
import os.path as osp
from datasets import load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from build_bm25_index import *
import tomllib

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--num_processes', type=int, default=30)
    parser.add_argument('--download_repo', action='store_true', 
                        help='Whether to download the codebase to `repo_path` before indexing.')
    parser.add_argument('--repo_path', type=str, default='playground/build_graph', 
                        help='The directory where you plan to pull or have already pulled the codebase.')
    parser.add_argument('--index_dir', type=str, default='index_data', 
                        help='The base directory where the generated graph index will be saved.')
    parser.add_argument('--instance_id_path', type=str, default='', 
                        help='Path to a file containing a list of selected instance IDs.')
    args = parser.parse_args()

    args.download_repo = True
    args.index_dir = '/Users/zhangmengqi/Documents/PhD/Working Documents/DaLIC_paper/validation_experiments/LocAgent/bm25_index'
    args.instance_id_path = '/Users/zhangmengqi/Documents/PhD/Working Documents/DaLIC_paper/validation_experiments/LocAgent/LocAgent/config.toml'
    
    dataset_name = args.dataset.split('/')[-1]
    args.index_dir = f'{args.index_dir}/{dataset_name}/BM25_index/'
    os.makedirs(args.index_dir, exist_ok=True)

    # load selected repo instance id and instance_data
    if args.download_repo:
        selected_instance_data = {}
        bench_data = load_dataset(args.dataset, split=args.split)
        if args.instance_id_path and osp.exists(args.instance_id_path):
            with open(args.instance_id_path, 'rb') as f:
                repo_folders = tomllib.load(f)['selected_ids']
            for instance in bench_data:
                if instance['instance_id'] in repo_folders:
                    selected_instance_data[instance['instance_id']] = instance
        else:
            repo_folders = []
            for instance in bench_data:
                repo_folders.append(instance['instance_id'])
                selected_instance_data[instance['instance_id']] = instance
    else:
        if args.instance_id_path and osp.exists(args.instance_id_path):
            with open(args.instance_id_path, 'rb') as f:
                repo_folders = tomllib.load(f)['selected_ids']
        else:
            repo_folders = list_folders(args.repo_path)
        selected_instance_data = None

    os.makedirs(args.repo_path, exist_ok=True)

    # Create a shared queue and add repositories to it
    manager = mp.Manager()
    queue = manager.Queue()
    for repo in repo_folders:
        queue.put(repo)

    start_time = time.time()

    mp.spawn(
        run,
        nprocs=args.num_processes,
        args=(queue, args.repo_path, args.index_dir,
              args.download_repo, selected_instance_data),
        join=True
    )

    end_time = time.time()
    print(f'Total Execution time = {end_time - start_time:.3f}s')