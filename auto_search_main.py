import argparse
import os
import json
import logging
import logging.handlers
import time
import toml
from queue import Queue
from typing import List
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset

from util.runtime.execute_ipython import execute_ipython
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
)
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
)
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError
from evaluation.eval_metric import filtered_instances


from time import sleep
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function)
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset


def get_task_instruction(instance: dict, task: str = 'auto_search', include_pr=False, include_hint=False):
    output_format = None
    instruction = ""
    
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = auto_search.TASK_INSTRUECTION.format(
            package_name=instance['instance_id'].split('_')[0]
        )
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC
        
    else:
        return None

    instruction += task_description
        
    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description = '\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )
    
    if output_format:
        instruction += output_format
    
    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    # NOTE: You can actually set slightly different instruction for different task
    # instruction += AGENT_CLS_TO_INST_SUFFIX
    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        tools = None,
                        traj_data=None,
                        temp=1.0,
                        max_iteration_num=20,
                        use_function_calling=True):
    if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower() 
    #             #   or model_name=='azure/gpt-4o' 
    #             #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                ):
        use_function_calling = False
        
    # for LLM which do not support function calling
    if not use_function_calling:
        # 转换message
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
        
    cur_interation_num = 0
    last_message = None
    finish = False
    while not finish:
        cur_interation_num += 1
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })

        try:
            # new conversation
            if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()):
                messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    temperature=temp, top_p=0.8, repetition_penalty=1.05, 
                    messages=messages,
                    stop=NON_FNCALL_STOP_WORDS
                )
            elif tools:
                response = litellm.completion(
                    model=model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                )
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            result_queue.put({'error': str(e), 'type': 'BadRequestError'})
            return
        
        if last_message and response.choices[0].message.content == last_message:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue
        
        raw_response = deepcopy(response)
        # logging.info('response.choices[0].message')
        if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()
                      or 'deepseek' in model_name
                      ):
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except:
                logging.info('convert none fncall messages failed.')
                continue 
                
        last_message = response.choices[0].message.content
        print(response.choices[0].message)
        messages.append(convert_to_json(raw_response.choices[0].message))
        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens  
            
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        for action in actions:
            logging.debug(action.action_type)
            if action.action_type == ActionType.FINISH:
                final_output = action.thought
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)
                
                logging.info("OBSERVATION:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
            else:
                logging.warning('Error Action!')
                # return

    # save traj
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    }
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def build_empty_loc_result(bug: dict):
    return {
        "instance_id": bug["instance_id"],
        "found_files": [[]],
        "found_modules": [[]],
        "found_entities": [[]],
        "raw_output_loc": [],
        "meta_data": {
            'repo': bug['repo'],
            'base_commit': bug['base_commit'],
            'problem_statement': bug['problem_statement'],
            'patch': bug['patch'],
        }
    }


def append_empty_loc_result(bug: dict, output_file_lock, output_file: str):
    with output_file_lock:
        append_to_jsonl(build_empty_loc_result(bug), output_file)


def run_localize_issue(rank, args, bug, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    instance_id = bug["instance_id"]
    prompt_manager = PromptManager(
        prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
        agent_skills_docs=LocationToolsRequirement.documentation,
    )

    try:
        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        set_current_issue(instance_data=bug, rank=rank)

        # loc result
        raw_output_loc = []
        loc_trajs = {'trajs': []}
        total_prompt_tokens, total_completion_tokens = 0, 0

        for _ in range(args.num_samples):
            logger.info("=" * 60)
            logger.info(f"==== rank {rank} begin localizing {instance_id} ====")
            max_attempt_num = args.max_attempt_num
            while max_attempt_num:
                logger.info("=" * 60)
                logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                loc_start_time = time.time()
                try:
                    """
                    Basic instructions:
                        - CodeAct instruction
                        - Few-shot Examples
                    """
                    if args.use_function_calling:
                        system_prompt = function_calling.SYSTEM_PROMPT
                        # system_prompt = CLAUDE_THINKING_INSTRUCTION
                    else:
                        system_prompt = prompt_manager.system_message
                        
                    messages: list[dict] = [{
                        "role": "system",
                        "content": system_prompt
                    }]
                        
                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

                    logger.info(f"==== {instance_id} start auto search ====")
                    messages.append({
                        "role": "user",
                        "content": get_task_instruction(bug, include_pr=True, include_hint=True),
                    })
                    
                    tools = None
                    # if args.use_function_calling:
                    tools = function_calling.get_tools(
                        codeact_enable_search_keyword=True,
                        codeact_enable_search_entity=True,
                        codeact_enable_tree_structure_traverser=True,
                        simple_desc = args.simple_desc,
                    )
                    result_queue = Queue()
                    logger.info(
                        f"{instance_id} running auto search inside issue process rank {rank}."
                    )
                    auto_search_process(
                        result_queue=result_queue,
                        model_name=args.model,
                        messages=messages,
                        fake_user_msg=auto_search.FAKE_USER_MSG_FOR_LOC,
                        temp=1,
                        tools=tools,
                        use_function_calling=args.use_function_calling,
                    )
                    
                    # loc_result, messages, traj_data = result_queue.get()
                    result = result_queue.get()
                    if isinstance(result, dict) and 'error' in result and result['type'] == 'BadRequestError':
                        raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                        # print(f"Error occurred in subprocess: {result['error']}")
                    else:
                        loc_result, messages, traj_data = result
                        
                except litellm.BadRequestError as e:
                    logger.warning(f'{e}. BadRequestError Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue
                except APITimeoutError:
                    logger.warning(f"APITimeoutError. Try again.")
                    sleep(10)
                    continue
                except litellm.exceptions.ContextWindowExceededError as e:
                    logger.warning(f'{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue

                loc_end_time = time.time()
                if not loc_result:
                    continue # empty result

                total_prompt_tokens += traj_data['usage']['prompt_tokens']
                total_completion_tokens += traj_data['usage']['completion_tokens']
                traj_data['time'] = loc_end_time - loc_start_time
                loc_trajs['trajs'].append(traj_data)

                # generate correct output or finish last attempt
                raw_output_loc.append(loc_result)
                break

        if not raw_output_loc:
            # loc generalization failed
            logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
            append_empty_loc_result(bug, output_file_lock, args.output_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            # all_valid_files = get_all_valid_files()
            all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                instance_id, raw_output_loc
            )
            
            loc_res = {
                "instance_id": instance_id,
                "found_files": all_found_files,
                "found_modules": all_found_modules,
                "found_entities": all_found_entities,
                "raw_output_loc": raw_output_loc,
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug['base_commit'],
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch'],
                    # 'gt_file_changes': gt_file_changes
                }
            }
            
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)

            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                append_to_jsonl(loc_res, traj_file)
    except Exception:
        logger.exception(f"Unhandled error while localizing {instance_id}. Saving empty output.")
        append_empty_loc_result(bug, output_file_lock, args.output_file)
    finally:
        try:
            reset_current_issue()
        except Exception:
            logger.exception(f"Failed to reset current issue for {instance_id}.")


def localize(args):
    bench_data = load_dataset(args.dataset, split=args.split)
    bench_tests = filter_dataset(bench_data, 'instance_id', args.used_list)
    if args.eval_n_limit:
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        bench_tests = bench_tests.select(range(0, eval_n_limit))
        logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    manager = mp.Manager()
    output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()

    # collect processed instances
    processed_instance = []
    if os.path.exists(args.output_file):
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        locs = load_jsonl(args.output_file)        
        if args.rerun_empty_location:
            traj_datas = load_jsonl(traj_file)
            backup_loc_output = backup_file(args.output_file)
            backup_traj_output = backup_file(traj_file)
            clear_file(args.output_file)
            clear_file(traj_file)
            for loc in locs:
                if loc['found_files'] != [[]]:
                    append_to_jsonl(loc, args.output_file)
                    processed_instance.append(loc['instance_id'])
                    
            for loc_traj in traj_datas:
                if loc_traj['found_files'] != [[]]:
                    append_to_jsonl(loc_traj, traj_file)
        else:
            processed_instance = [loc['instance_id'] for loc in locs]
    
    pending_bugs = []
    for bug in bench_tests:
        instance_id = bug["instance_id"]
        if instance_id in processed_instance:
        # if instance_id in processed_instance or instance_id in filtered_instances:
            print(f"instance {instance_id} has already been processed, skip.")
        else:
            pending_bugs.append(bug)

    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    try:
        num_bugs = len(pending_bugs)
        max_parallel = (
            min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs
        )
        if max_parallel <= 0:
            return

        ctx = mp.get_context("spawn")
        active_processes = {}
        available_ranks = list(range(max_parallel))

        while pending_bugs or active_processes:
            while pending_bugs and available_ranks:
                rank = available_ranks.pop(0)
                bug = pending_bugs.pop(0)
                process = ctx.Process(
                    target=run_localize_issue,
                    kwargs={
                        "rank": rank,
                        "args": args,
                        "bug": bug,
                        "log_queue": log_queue,
                        "output_file_lock": output_file_lock,
                        "traj_file_lock": traj_file_lock,
                    },
                )
                process.start()
                active_processes[rank] = {
                    "process": process,
                    "bug": bug,
                    "start_time": time.time(),
                }

            time.sleep(1)
            for rank, proc_info in list(active_processes.items()):
                process = proc_info["process"]
                bug = proc_info["bug"]
                process.join(timeout=0)

                if not process.is_alive():
                    if process.exitcode not in (0, None):
                        logging.warning(
                            f"{bug['instance_id']} exited with code {process.exitcode}. Saving empty output."
                        )
                        append_empty_loc_result(bug, output_file_lock, args.output_file)
                    del active_processes[rank]
                    available_ranks.append(rank)
                    available_ranks.sort()
                    continue

                elapsed = time.time() - proc_info["start_time"]
                if elapsed > args.timeout:
                    logging.warning(
                        f"{bug['instance_id']} exceeded top-level timeout ({args.timeout}s). Terminating issue process."
                    )
                    process.terminate()
                    process.join()
                    append_empty_loc_result(bug, output_file_lock, args.output_file)
                    del active_processes[rank]
                    available_ranks.append(rank)
                    available_ranks.sort()
    finally:
        queue_listener.stop()
    
    if args.rerun_empty_location:
        try:
            delete_file(backup_loc_output)
            delete_file(backup_traj_output)
        except:
            return


def merge(args):
    args.merge_file = os.path.join(args.output_folder, 'merged_' + os.path.basename(args.output_file))
    
    if args.ranking_method == 'mrr':
        args.merge_file = args.merge_file.replace('.jsonl', f'_{args.ranking_method}.jsonl')
        
    clear_file(args.merge_file)
    with open(args.output_file, 'r') as file:
        for line in file:
            loc_data = json.loads(line)
            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_modules'] = []
                loc_data['found_entities'] = []
            else:
                loc_data['found_files'] = loc_data['found_files']
                loc_data['found_modules'] = loc_data['found_modules']
                loc_data['found_entities'] = loc_data['found_entities']
                ranked_files, ranked_modules, ranked_funcs = merge_sample_locations(loc_data['found_files'], 
                                                                    loc_data['found_modules'],
                                                                    loc_data['found_entities'],
                                                                    ranking_method=args.ranking_method,
                                                                    )
                loc_data['found_files'] = ranked_files
                loc_data['found_modules'] = ranked_modules
                loc_data['found_entities'] = ranked_funcs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--ranking_method", type=str, default='mrr',
                        choices=['mrr', 'majority'])
    
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--used_list", type=str, default='selected_ids')
    
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    
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
    parser.add_argument("--use_function_calling", action="store_true",
                        help='Enable function calling features of LLMs. If disabled, codeact will be used to support function calling.')
    parser.add_argument("--simple_desc", action="store_true", 
                        help="Use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.")
    
    parser.add_argument("--max_attempt_num", type=int, default=1, 
                        help='Only use in generating training trajectories.')
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_processes", type=int, default=-1)
    
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
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
    
    if args.localize:
        localize(args)
    
    
    if args.merge:
        merge(args)


if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()
    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
