from auto_search_main import *

def get_arg():
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
    parser.add_argument("--output_folder", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
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
        ]
    )
    # 是否启用 LLM 原生 Function Calling 能力
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    # 是否使用简化版的工具描述（Claude 等强模型建议设为 False）
    parser.add_argument("--simple_desc", action="store_true", 
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    # 遇到框架报错或超时时重新尝试生成的最大次数
    parser.add_argument("--max_attempt_num", type=int, default=1, 
                        help='Only use in generating training trajectories.')
    # 每次为同一个 issue 生成的定位结果采样数，后续通过 --merge 聚合
    parser.add_argument("--num_samples", type=int, default=1)
    # 指定多进程并行执行的工作进程数（-1 代表全量并行）
    parser.add_argument("--num_processes", type=int, default=-1)
    
    # 设置 logging 的日志级别
    parser.add_argument("--log_level", type=str, default='INFO')
    # 单个 issue 并行进程最多允许运行的时间（秒）
    parser.add_argument("--timeout", type=int, default=900)
    # 重新运行之前未能找到合法预测文件的 instance_id
    parser.add_argument("--rerun_empty_location", action="store_true")
    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )
    return args

if __name__ == "__main__":
    arg = get_arg()
    if not arg.localize:
        arg.localize = True

    start_time = time.time()
    localize(arg)
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))