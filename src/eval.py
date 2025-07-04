import json
import os
from vllm import LLM, SamplingParams
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the extract parameters model on test data.")
    parser.add_argument("--model", type=str, default="/data01/xushuai/code/output/agent/agent_32b_v23", help="Path to the model.")
    parser.add_argument("--test-data", type=str, default="/data01/xushuai/code/data/general-raw/extract_params_test.jsonl", help="Path to the test data.")
    parser.add_argument("--qwen3", action="store_true", help="Use Qwen3 model.")
    return parser.parse_args()


def find_model_path():
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


def is_equal(pred, tgt, eval_task):
    EVAL_MAP = {
        "dimension detail": is_equal_dimension_detail,
        "groupby": is_equal_group_by,
        "intent in-scene": is_equal_intent,
        "intent out-scene": is_equal_intent,
        "metric detail": is_equal_metric_detail,
        "metric": is_equal_metric,
        "orderby": is_equal_order_by,
        "time attribution": is_equal_time_attr,
        "time dimensions": is_equal_time_dimension,
        "time range": is_equal_time_range,
        "where": is_equal_where,
    }
    return EVAL_MAP[eval_task](pred, tgt)


def is_equal_dimension_detail(pred, tgt):
    return pred == tgt


def is_equal_group_by(pred, tgt):
    if pred == tgt:
        return True
    if "groupBys" not in pred:
        return False
    pred, tgt = set(pred["groupBys"]), set(tgt["groupBys"])
    return pred == tgt


def is_equal_intent(pred, tgt):
    if pred == tgt:
        return True
    if "intent_list" not in pred:
        return False
    pred_intents, tgt_intents = set(pred["intent_list"]), set(tgt["intent_list"])
    if pred_intents != tgt_intents:
        return False
    if "fist_question" in pred:
        if "fist_question" not in tgt or pred["fist_question"] != tgt["fist_question"]:
            return False
    return True


def is_equal_metric_detail(pred, tgt):
    return pred == tgt


def is_equal_metric(pred, tgt):
    if pred == tgt:
        return True
    if "metrics" not in pred:
        return False
    pred_metrics, tgt_metrics = set(pred["metrics"]), set(tgt["metrics"])
    return pred_metrics == tgt_metrics


def is_equal_order_by(pred, tgt):
    if pred == tgt:
        return True
    if "orderBys" not in pred:
        return False
    order_by_pred, order_by_tgt = set(pred["orderBys"]), set(tgt["orderBys"])
    if order_by_pred != order_by_tgt:
        return False
    if "limit" in tgt:
        if "limit" not in pred or pred["limit"] != tgt["limit"]:
            return False
    return True


def is_equal_time_attr(pred, tgt):
    if pred == tgt:
        return True
    if "baseTime" not in pred or "compareTime" not in pred:
        return False
    if tgt == {} and pred != {}:
        return False
    base_pred, base_tgt = pred["baseTime"], tgt["baseTime"]
    compare_pred, compare_tgt = pred["compareTime"], tgt["compareTime"]
    if base_pred != base_tgt:
        return False
    if compare_pred != compare_tgt:
        return False
    return True


def is_equal_time_dimension(pred, tgt):
    if pred == tgt:
        return True
    if "timeDimension" not in pred:
        return False
    pred_time, tgt_time = set(pred["timeDimension"]), set(tgt["timeDimension"])
    return pred_time == tgt_time


def is_equal_time_range(pred, tgt):
    if pred == tgt:
        return True
    if "timeStartFunction" not in pred or "timeEndFunction" not in pred:
        return False
    if tgt == {} and pred != {}:
        return False
    pred_start, pred_end = pred["timeStartFunction"], pred["timeEndFunction"]
    tgt_start, tgt_end = tgt["timeStartFunction"], tgt["timeEndFunction"]
    if pred_start != tgt_start or pred_end != tgt_end:
        return False
    return True


def is_equal_where(pred, tgt):
    if pred == tgt:
        return True
    if "where" not in pred:
        return False
    if "AND" in tgt and "OR" in tgt:
        return False  # cannot handle mixed AND/OR conditions
    
    pred_where, tgt_where = pred["where"], tgt["where"]
    pred_where, tgt_where = pred_where.strip(), tgt_where.strip()
    
    if not pred_where or not tgt_where:
        return pred_where == tgt_where
    
    pred = quote_sql_values(pred_where)
    tgt = quote_sql_values(tgt_where)

    if len(pred) != len(tgt):
        return False
    for p in pred:
        if p not in tgt:
            return False
    return True


def quote_sql_values(where_clause: str) -> str:
    where_clause = where_clause.strip()
    cond_pattern = r"(\w+)\s*(=|!=|<>|<=|>=|<|>|IN)\s*(\([^)]+\)|[^()'\" \t\n\r\f\v]+(?:\s[^()'\" \t\n\r\f\v]+)*)"
    logic_pattern = r"\s+(AND|OR)\s+"
    parts = re.split(f"({logic_pattern})", where_clause.strip(), flags=re.IGNORECASE)
    results = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(cond_pattern, part, flags=re.IGNORECASE):
            match = re.match(cond_pattern, part, flags=re.IGNORECASE)
            key, op, value = match.groups()
            if op == "IN":
                value = value.replace("(", "").replace(")", "").strip()
                value = [v.strip().strip("'") for v in value.split(",")]
                value = sorted(value)  # Sort for consistent comparison
                value = f"({', '.join(value)})"
            results.append({
                'key': key,
                'op': op.upper(),
                'value': value
            })
            i += 2
        else:
            i += 1
    return results


def main(args):
    with open(args.test_data) as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if args.qwen3:
        for d in test_data:
            d["messages"][0]["content"] += " /no_think"
    messages = [d["messages"][:-1] for d in test_data]
    
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)
    model_path = find_model_path()
    tensor_parallel_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) or 1
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=2048, tensor_parallel_size=tensor_parallel_size)
    outputs = llm.chat(messages, sampling_params)
    
    log_dir = os.path.join(os.path.dirname(model_path), "eval")
    os.makedirs(log_dir, exist_ok=True)
    log_files = {}
    for d in test_data:
        if d["metadata"]["description"] not in log_files:
            log_files[d["metadata"]["description"]] = open(os.path.join(log_dir, f"{d['metadata']['description']}.log"), "w")

    results = {}
    for d, output in zip(test_data, outputs):
        result = output.outputs[0].text.strip()
        # print(result)
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        tgt = d["messages"][-1]["content"].split("</think>")[-1].strip()
        if d["metadata"]["description"] not in results:
            results[d["metadata"]["description"]] = {"correct": 0, "total": 0}
        try:
            result, tgt = json.loads(result), json.loads(tgt)
            if is_equal(result, tgt, d["metadata"]["description"]):
                results[d["metadata"]["description"]]["correct"] += 1
            else:
                log_files[d["metadata"]["description"]].write(f"Prompt: {d['messages'][0]['content']}\n")
                log_files[d["metadata"]["description"]].write(f"Result: {json.dumps(result, ensure_ascii=False)}\n")
                log_files[d["metadata"]["description"]].write(f"Target: {json.dumps(tgt, ensure_ascii=False)}\n")
                log_files[d["metadata"]["description"]].write("\n")
        except Exception:
            pass
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    args = parse_args()
    main(args)
