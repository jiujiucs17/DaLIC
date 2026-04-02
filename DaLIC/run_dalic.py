import os
import sys
import pandas as pd
from tqdm import tqdm
import pickle
import logging

# Suppress LiteLLM verbose output in console
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["DALIC_SILENCE_LITELLM_STDOUT"] = "1"
import litellm
litellm.set_verbose=False

# Set environment variables for LLM access
os.environ["GRAPH_INDEX_DIR"] = "/Users/zhangmengqi/Documents/PhD/Working Documents/DaLIC_paper/validation_experiments/LocAgent/graph_index"
os.environ["BM25_INDEX_DIR"] = "/Users/zhangmengqi/Documents/PhD/Working Documents/DaLIC_paper/validation_experiments/LocAgent/bm25_index"
os.environ["LOCAL_REPO_CACHE"] = "/Users/zhangmengqi/Documents/PhD/Working Documents/DaLIC_paper/validation_experiments/LocAgent/repo_cache"
os.environ["HOSTED_VLLM_API_BASE"] = "https://ezo2psrlnu5b3q-8000.proxy.runpod.net/v1"
os.environ["HOSTED_VLLM_API_KEY"] = "sk-352cab55f6fd755ca0c2011514de88677101c291e2bba77abeef2cc92c1fe6ea"


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.eval_metric import evaluate_results
from auto_search_main import *
from evaluation.eval_metric import evaluate_results

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}
selected_ids = [
     "scikit-learn__scikit-learn-24145",
     "pandas-dev__pandas-29944",
     "scikit-learn__scikit-learn-16948",
     "scikit-learn__scikit-learn-17443",
     "scikit-learn__scikit-learn-14800",
     "scikit-learn__scikit-learn-30327",
     "scikit-learn__scikit-learn-30128",
     "scikit-learn__scikit-learn-28678",
     "pandas-dev__pandas-59900",
     "pandas-dev__pandas-60526",
     "pandas-dev__pandas-60518",
     "pandas-dev__pandas-60457", 
    #  "pandas-dev__pandas-60415", # frequently exceeds context window
     "pandas-dev__pandas-60277",
     "pandas-dev__pandas-60247",
     "pandas-dev__pandas-60187",
     "sympy__sympy-27325",
     "matplotlib__matplotlib-29133",
     "matplotlib__matplotlib-29265",
     "numpy__numpy-17394",
]
# eval with dataset
k_values_list = [
    [1, 3, 5, 10, -1],
    [1, 3, 5, 10, -1],
    [1, 3, 5, 10, -1]
]
config = {
    "original": (False, False),
    "with_dalic": (True, False),
    "with_dalic_data_deps": (True, True),
}
output_folder_root = os.path.join(os.path.dirname(__file__), "outputs")

def save_eval_results_txt(eval_results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result_name, result_df in eval_results.items():
            f.write(f"===== {result_name} =====\n")
            if isinstance(result_df, pd.DataFrame):
                f.write(result_df.to_string())
            else:
                f.write(str(result_df))
            f.write("\n\n")

def get_arg(output_folder = None,
            use_dalic=False,
            use_data_deps=False):
    parser = argparse.ArgumentParser()
    # 是否开启“代码定位”（localization）主流程
    parser.add_argument("--localize", action="store_true")
    # 是否开启“合并”流程。在使用多次采样（--num_samples > 1）生成了多个预测结果后，加上此参数会将多次的结果做合并和重排序
    parser.add_argument("--merge", action="store_true")
    # 是否使用 Few-shot examples（少样本示例），在提示词中给 LLM 提供示例
    parser.add_argument("--use_example", action="store_true")
    # 用于在合并多次采样（--merge）时采用的排序方法（'mrr' 或 'majority'），仅在merge选项开启时有效
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    # 指定 HuggingFace 数据集的名称及路径
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    # 指定要使用的数据集划分
    parser.add_argument("--split", type=str, default="test")
    # 最多评估的样本数量，0表示评估所有样本
    parser.add_argument("--eval_n_limit", type=int, default=0)
    # 用于在 config.toml 中查找要过滤处理的特定列表名称，只处理该列表里的 instance_id
    parser.add_argument("--used_list", type=str, default='selected_ids')
    
    # 指定用来存放所有输出文件（如日志、追踪轨迹、结果 JSONL）的目录
    parser.add_argument("--output_folder", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
    # 用来存放每次预测的未合并原始结果的文件名
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    # 用来存放运行 --merge 之后合并重排序的最终文件的名字
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    
    # 指定要使用的 LLM 模型名称
    parser.add_argument(
        "--model", type=str,
        default="openai/gpt-4o-2024-05-13",
        choices=["gpt-4o", 
                 "azure/gpt-4o", "openai/gpt-4o-2024-05-13",
                 "deepseek/deepseek-chat", "deepseek-ai/DeepSeek-R1",
                 "litellm_proxy/claude-3-5-sonnet-20241022", "litellm_proxy/gpt-4o-2024-05-13", "litellm_proxy/o3-mini-2025-01-31",
                 # fine-tuned model
                 "openai/qwen-7B", "openai/qwen-7B-128k", "openai/ft-qwen-7B", "openai/ft-qwen-7B-128k",
                 "openai/qwen-32B", "openai/qwen-32B-128k", "openai/ft-qwen-32B", "openai/ft-qwen-32B-128k",
                 "hosted_vllm/czlll/Qwen2.5-Coder-7B-CL", "openai/czlll/Qwen2.5-Coder-32B-CL"
        ]
    )
    # 是否启用 LLM 原生 Function Calling 能力
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    # 是否使用简化版的工具描述（Claude 等强模型建议设为 False）
    parser.add_argument("--simple_desc", action="store_true", 
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    # 遇到框架报错或超时时重新尝试生成的最大次数
    parser.add_argument("--max_attempt_num", type=int, default=2, 
                        help='Only use in generating training trajectories.')
    # 每次为同一个 issue 生成的定位结果采样数，后续通过 --merge 聚合
    parser.add_argument("--num_samples", type=int, default=1)
    # 指定多进程并行执行的工作进程数（-1 代表全量并行）
    parser.add_argument("--num_processes", type=int, default=-1)
    
    # 设置 logging 的日志级别
    parser.add_argument("--log_level", type=str, default='INFO')
    # 单个 issue 并行进程最多允许运行的时间（秒）
    parser.add_argument("--timeout", type=int, default=1500)
    # 重新运行之前未能找到合法预测文件的 instance_id
    parser.add_argument("--rerun_empty_location", action="store_true")

    # DaLIC related arguments
    parser.add_argument("--use_dalic", action="store_true", help="Whether to use DaLIC for localization.")
    parser.add_argument("--use_data_deps", action="store_true", help="Whether to include data dependencies in the context for localization.")
    args = parser.parse_args()

    # set arguments according to the parameters of the function
    if output_folder is not None:
        args.output_folder = output_folder
    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    args.use_dalic = use_dalic# Whether to use DaLIC for localization.
    args.use_data_deps = use_data_deps  # Whether to include data dependencies in the context for

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            # logging.StreamHandler()
        ],
        force=True
    )
    return args

if __name__ == "__main__":
    # print start time
    # print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    # for config_name, (use_dalic, use_data_deps) in config.items():
    #     for i in tqdm(range(5), desc=f"Running {config_name}"):
    #         output_folder_runtime = os.path.join(output_folder_root, f"{config_name}_run_{i+1}")
    #         arg = get_arg(output_folder=output_folder_runtime, 
    #                       use_dalic=use_dalic, 
    #                       use_data_deps=use_data_deps)
    #         arg.localize = True
    #         arg.dataset = "JJcs17/Loc-Bench-add_fixed_commit"
    #         arg.model = "hosted_vllm/czlll/Qwen2.5-Coder-7B-CL"
    #         arg.num_processes = 5

    #         # write the arguments
    #         with open(f"{arg.output_folder}/args.json", "w") as f:
    #             json.dump(vars(arg), f, indent=4)
    #             print(f"Finished running {config_name} for 5 times.")

    #         start_time = time.time()
    #         localize(arg)
    #         merge(arg)
    #         end_time = time.time()
    #         logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))

    # evaluate results
    eval_results = {}
    for config_name in config.keys():
        config_eval_result = []
        for i in tqdm(range(5), desc=f"Evaluating {config_name}"):
            output_folder_runtime = os.path.join(output_folder_root, f"{config_name}_run_{i+1}")
            locagent_loc_file = os.path.join(output_folder_runtime, "merged_loc_outputs_mrr.jsonl")
            locagent_res = evaluate_results(locagent_loc_file,
                        level2key_dict,
                        dataset='JJcs17/Loc-Bench-add_fixed_commit',
                        metrics=['precision', 'recall'],
                        selected_list=selected_ids,
                        k_values_list = k_values_list
                        )
            run_eval_df = pd.DataFrame(locagent_res).T
            run_eval_df.index.name = "level"
            eval_results[f"{config_name}_run_{i+1}"] = run_eval_df
            config_eval_result.append(run_eval_df)
        # calculate the average result for the 5 runs of the same config
        avg_eval_df = pd.concat(config_eval_result).groupby(level=0).mean()
        avg_eval_df.index.name = "level"
        eval_results[f"{config_name}_average"] = avg_eval_df
        print(f"Finished evaluating results for {config_name}.")

    eval_results_file = os.path.join(output_folder_root, "eval_results.txt")
    save_eval_results_txt(eval_results, eval_results_file)
    print(f"Saved evaluation results to {eval_results_file}.")




    # arg = get_arg(use_data_deps=True)
    # if not arg.localize:
    #     arg.localize = True
    # arg.dataset = "JJcs17/Loc-Bench-add_fixed_commit"
    # arg.model = "hosted_vllm/czlll/Qwen2.5-Coder-7B-CL"
    # arg.num_processes = 5
    # start_time = time.time()
    # localize(arg)
    # merge(arg)
    # end_time = time.time()
    # logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
