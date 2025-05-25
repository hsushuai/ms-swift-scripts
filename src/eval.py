import json
import os
from vllm import LLM, SamplingParams
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the extract parameters model on test data.")
    parser.add_argument("--model", type=str, default="/data01/xushuai/code/output/general/general_batch_24_acc_4_v1", help="Path to the model.")
    parser.add_argument("--test-data", type=str, default="/data01/xushuai/code/data/general-raw/extract_params_test.jsonl", help="Path to the test data.")
    parser.add_argument("--qwen3", action="store_true", help="Use Qwen3 model.")
    return parser.parse_args()


def find_model_path(model):
    if any(d.endswith(".safetensors") for d in os.listdir(args.model)):
        model_path = args.model
    else:
        checkpoint_dirs = [
            d for d in os.listdir(args.model)
            if d.startswith("checkpoint") and os.path.isdir(os.path.join(args.model, d))
        ]

        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint directories found in {args.model}")

        # extract the latest checkpoint number
        def extract_step(d):
            match = re.search(r"checkpoint-(\d+)", d)
            return int(match.group(1)) if match else -1

        latest_ckpt = max(checkpoint_dirs, key=extract_step)
        model_path = os.path.join(args.model, latest_ckpt)
    return model_path


def main(args):
    with open(args.test_data) as f:
        test_data = [json.loads(line) for line in f.readlines()]
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)
    model_path = find_model_path(args.model)
    tensor_parallel_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) or 1
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=2048, tensor_parallel_size=tensor_parallel_size)
    # outputs = llm.generate([d["messages"][0]["content"] for d in test_data], sampling_params)
    if args.qwen3:
        for d in test_data:
            d["messages"][0]["content"] += " /no_think"
    outputs = llm.chat([d["messages"] for d in test_data], sampling_params)
    log_dir = os.path.join(os.path.dirname(model_path), "eval")
    os.makedirs(log_dir, exist_ok=True)
    log_files = {}
    for d in test_data:
        if d["metadata"]["description"] not in log_files:
            log_files[d["metadata"]["description"]] = open(os.path.join(log_dir, f"{d['metadata']['description']}.log"), "w")

    results = {}
    for d, output in zip(test_data, outputs):
        result = output.outputs[0].text.strip()
        print(result)
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        tgt = d["messages"][-1]["content"].split("</think>")[-1].strip()
        if d["metadata"]["description"] not in results:
            results[d["metadata"]["description"]] = {"correct": 0, "total": 0}
        try:
            result, tgt = json.loads(result), json.loads(tgt)
        except json.JSONDecodeError:
            pass
        if result == tgt:
            results[d["metadata"]["description"]]["correct"] += 1
        else:
            log_files[d["metadata"]["description"]].write(f"Prompt: {d['messages'][0]['content']}\n")
            log_files[d["metadata"]["description"]].write(f"Result: {json.dumps(result)}\n")
            log_files[d["metadata"]["description"]].write(f"Target: {json.dumps(tgt)}\n")
            log_files[d["metadata"]["description"]].write("\n")
        results[d["metadata"]["description"]]["total"] += 1

    results_text = ""
    for k, v in results.items():
        text = f"{k}: {v['correct']}/{v['total']} = {v['correct'] / v['total']:.2%}"
        print(text)
        results_text += text + "\n"
        log_files[k].write(f"{k}: {v['correct']}/{v['total']} = {v['correct'] / v['total']:.2%}\n")
        log_files[k].close()
    
    with open(os.path.join(log_dir, "all_results.log"), "w") as f:
        f.write(results_text)
    
    print(f"Evaluation results saved to {log_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
