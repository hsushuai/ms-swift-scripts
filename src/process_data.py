import json
import pandas as pd
import yaml
import os
import random
import re
from datetime import datetime


def to_jsonl(filename, template, variables, process_fn=None, output=None, metadata=None):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename)
    results = []
    if process_fn:
        df = process_fn(df)
    for _, row in df.iterrows():
        if pd.isna(row[variables[-1]]):
            continue
        prompt = template.format(**{k: str(row[k]).strip() for k in variables[:-1]})
        result = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<think>\n\n</think>\n\n{row[variables[-1]]}"}]}
        if metadata:
            result["metadata"] = metadata
        results.append(result)
    file_type = filename.split("/")[-1].split(".")[-1]
    output = filename.replace("/raw/", "/processed/").replace(file_type, "jsonl") if not output else output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Processed {filename} to {output} with {len(results)} records.")
    show_message(results)


def dims_formal(a):
    if pd.isna(a):
        return "()"
    pattern = re.compile(r"(\w+): \[(.*?)\]")
    matches = pattern.findall(a)
    output = ""
    for match in matches:
        description = match[0]
        values = match[1] if match[1] else ""
        output += f"(description: {description}, values: [{values}])\n"
    return output


def metrics_formal(a):
    if pd.isna(a):
        return "[]"
    return "[" + a + "]"
    

def process_train_intent(df):
    intents = ["连续问", "与召回相关", "非召回相关", "维度详情", "指标详情", "维度列表", "指标列表", "数据概述", "模型列表", "闲聊", "问指标", "问码值", "问知识", "需排序", "需对比", "需分析", "需同环比", "需对比全部指标", "需占比", "需分布"]
    df["question"] = df["问题"]
    df["metrics"] = df["metrics"].apply(lambda x: metrics_formal(x))
    df["dimensions"] = df["dimensions"].apply(lambda x: dims_formal(x))
    for idx, row in df.iterrows():
        res = {"intent_list": [intent for intent in intents if row[intent] == 1]}
        if row["连续问"] == 1:
            res["first_question"] = row["连续问首问题"]
        df.loc[idx, "ground_truth"] = json.dumps(res, ensure_ascii=False)
    return df


def merge_all_jsonl(filenames, output):
    all_data = []
    for filename in filenames:
        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))
        all_data.extend(data)
        print(f"Loaded {len(data)} records from {filename}.")
        show_message(data)
    with open(output, "w") as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Merged {len(all_data)} records into {output}.")


def show_message(messages):
    message = messages[random.randint(0, len(messages) - 1)]
    for m in message["messages"]:
        total_len = 40
        pre_len = (total_len - len(m["role"]) - 2) // 2
        suf_len = total_len - pre_len - len(m["role"]) - 2
        print("=" * pre_len + f" {m['role']} " + "=" * suf_len)
        print(m["content"])
    print("=" * total_len)


def process_time(df):
    def _process_time(x):
        if isinstance(x, datetime):
            dt = x
        elif isinstance(x, str):
            try:
                dt = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return x
        else:
            raise ValueError("Unsupported date format")
        return f"{dt.year}年{dt.month}月{dt.day}日"
    df["env"] = df["env"].apply(_process_time)
    return df


def convert_train_data():
    with open("template/prompt.yaml") as f:
        templates = yaml.safe_load(f)
    
    # Process extract parameters data
    data_files = [
        {"name": "维度详情", "path": "data/general-raw/raw/train/维度详情.xlsx", "variables": ["question", "env", "ground_truth"]},
        {"name": "指标详情", "path": "data/general-raw/raw/train/指标详情.xlsx", "variables": ["question", "env", "ground_truth"]},
        {"name": "group_by", "path": "data/general-raw/raw/train/group_by.xlsx", "variables": ["Input", "env", "groupby"]},
        {"name": "long_where_w_ans", "path": "data/general-raw/raw/train/long_where_w_ans.xlsx", "variables": ["question", "env", "response"]},
        {"name": "metric", "path": "data/general-raw/raw/train/metric.xlsx", "variables": ["question", "env", "ground_truth"]},
        {"name": "order_by", "path": "data/general-raw/raw/train/order_by.xlsx", "variables": ["query", "ground_truth"]},
        {"name": "time_dims", "path": "data/general-raw/raw/train/time_dims.xlsx", "variables": ["question", "env", "ground_truth"]},
        {"name": "time_range", "path": "data/general-raw/raw/train/time_range.xlsx", "variables": ["question", "env", "ground_truth"], "process_fn": process_time},
        {"name": "time归因", "path": "data/general-raw/raw/train/time归因.xlsx", "variables": ["question", "env", "ground_truth"], "process_fn": process_time},
        {"name": "where", "path": "data/general-raw/raw/train/where.xlsx", "variables": ["Input", "ground_truth"]},
        {"name": "pre意图识别", "path": "data/general-raw/raw/train/pre意图识别.xlsx", "variables": ["question", "metrics", "dimensions", "ground_truth"], "process_fn": process_train_intent}
    ]
    for data_file in data_files:
        template = templates[data_file["name"]].strip()
        filename = data_file["path"]
        variables = data_file["variables"]
        process_fn = data_file.get("process_fn")
        to_jsonl(filename, template, variables, process_fn)
    
    # Merge all JSONL files
    jsonl_files = [d["path"].replace("/raw/", "/processed/").replace(".xlsx", ".jsonl") for d in data_files]
    merge_all_jsonl(jsonl_files, "data/general-raw/extract_params_train.jsonl")
    print("Train data processing completed.")


def convert_test_data():
    with open("template/prompt.yaml") as f:
        templates = yaml.safe_load(f)
    
    # Process test data
    test_data_files = [
        {"name": "维度详情", "description": "metric detail", "path": "data/general-raw/raw/test/维度详情 - 测试集.csv", "variables": ["question", "env", "ground_truth"]},
        {"name": "指标详情", "description": "dimension detail", "path": "data/general-raw/raw/test/指标详情 - 测试集.csv", "variables": ["question", "env", "ground_truth"]},
        {"name": "group_by", "description": "groupby", "path": "data/general-raw/raw/test/提参测试集.csv", "variables": ["Input", "env", "groupby ground_truth"], "process_fn": process_test_groupby, "output": "data/processed/test/提参测试集_groupby.jsonl"},
        {"name": "where", "description": "where", "path": "data/general-raw/raw/test/提参测试集.csv", "variables": ["Input", "where ground_truth"], "output": "data/processed/test/提参测试集_where.jsonl"},
        {"name": "metric", "description": "metric", "path": "data/general-raw/raw/test/提参测试集.csv", "variables": ["question", "env", "metric ground_truth"], "process_fn": process_test_metric, "output": "data/processed/test/提参测试集_metric.jsonl"},
        {"name": "order_by", "description": "orderby", "path": "data/general-raw/raw/test/orderby - 测试集.csv", "variables": ["query", "ground_truth"]},
        {"name": "time_dims", "description": "time dimensions", "path": "data/general-raw/raw/test/时间提参 - time_dims测试数据.csv", "variables": ["question", "env", "ground_truth"]},
        {"name": "time_range", "description": "time range", "path": "data/general-raw/raw/test/时间提参 - time_range测试集.csv", "variables": ["question", "env", "Output"], "process_fn": process_time},
        {"name": "time归因", "description": "time attribution", "path": "data/general-raw/raw/test/时间提参 - time归因测试集.csv", "variables": ["question", "env", "ground_truth"], "process_fn": process_time},
        {"name": "pre意图识别", "description": "intent in-scene", "path": "data/general-raw/raw/test/pre意图识别数据 - 场景内测试集.csv", "variables": ["question", "metrics", "dimensions", "ground_truth"]},
        {"name": "pre意图识别", "description": "intent out-scene", "path": "data/general-raw/raw/test/pre意图识别数据 - 场景外测试集.csv", "variables": ["question", "metrics", "dimensions", "ground_truth"]},
    ]
    for data_file in test_data_files:
        template = templates[data_file["name"]].strip()
        filename = data_file["path"]
        variables = data_file["variables"]
        process_fn = data_file.get("process_fn")
        output = data_file.get("output")
        to_jsonl(filename, template, variables, process_fn, output, metadata={"description": data_file["description"]})
    
    # Merge all JSONL files
    test_dir = "data/general-raw/processed/test"
    jsonl_files = [os.path.join("data/general-raw/processed/test", f) for f in os.listdir(test_dir) if f.endswith(".jsonl")]
    merge_all_jsonl(jsonl_files, "data/general-raw/extract_params_test.jsonl")
    print("Test data processing completed.")


def process_test_groupby(df: pd.DataFrame) -> pd.DataFrame:
    df["env"] = df["Input"].apply(lambda x: "[Metric]:\n" + x.split("[Metric]:")[-1].strip())
    df["Input"] = df["Input"].apply(lambda x: x.split("[Metric]:")[0].strip())
    return df


def process_test_metric(df: pd.DataFrame) -> pd.DataFrame:
    df["env"] = df["Input"].apply(lambda x: "[Metric]:\n" + x.split("[Metric]:")[-1].split("[Dimension]")[0].strip())
    df["question"] = df["Input"].apply(lambda x: x.split("[Metric]:")[0].strip())
    return df

def process_test_intent(df: pd.DataFrame) -> pd.DataFrame:
    df["metrics"] = df["metrics"].apply(lambda x: metrics_formal(x))
    df["dimensions"] = df["dimensions"].apply(lambda x: dims_formal(x))
    return df


def process_think_empty(filename):
    with open(filename) as f:
        data = []
        num_think_empty = 0
        for line in f:
            data.append(json.loads(line))
    for d in data:
        messages = d["messages"]
        messages = [{"role": m["role"], "content": m["content"].strip()} for m in messages]
        if re.search(r"<think>\s*</think>", messages[-1]["content"]):
            num_think_empty += 1
            res = messages[-1]["content"].split("</think>")[-1].strip()
            messages[-1]["content"] = f"<think>\n\n</think>\n\n{res}"
            if "/no_think" not in messages[-2]["content"]:
                messages[-2]["content"] += " /no_think"
        d["messages"] = messages
    with open(filename, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Processed {filename} with {num_think_empty}/{len(data)} records containing empty <think> tags.")


def merge_train_data(data_dir, qwen3=False):
    filenames = [f"{data_dir}/{d}" for d in os.listdir(data_dir) if d != "train.jsonl" and d.endswith(".jsonl")]
    channels = set()
    for filename in filenames:
        # add channel info
        channel = filename.split("/")[-1].replace(".jsonl", "")
        channels.add(channel)
        with open(filename) as f:
            data = []
            for line in f:
                d = json.loads(line)
                d["channel"] = channel
                data.append(d)
        with open(filename, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if qwen3:
            process_think_empty(filename)
    merge_all_jsonl(filenames, f"{data_dir}/train.jsonl")
    len_str = sum(len(ch) + 2 + 1 for ch in channels) - 1
    len_pre_space = (len_str - len("Channels")) // 2
    print("=" * len_str)
    print(" " * len_pre_space + "Channels")
    print("-" * len_str)
    print(' '.join(f"'{ch}'" for ch in channels))
    print("=" * len_str)


if __name__ == "__main__":
    # convert_train_data()
    # convert_test_data()
    # process_think_empty("data/condense-3/condense.jsonl")
    merge_train_data("data/agent-23", qwen3=True)
