import json
import os
import re


def is_dirty(messages):
    num_open = re.findall("</think>", messages[-1]["content"], re.DOTALL)
    num_close = re.findall("<think>", messages[-1]["content"], re.DOTALL)
    num_think = re.findall(r"<think>.*?</think>", messages[-1]["content"], re.DOTALL)
    if not (len(num_think) == len(num_open) == len(num_close) == 1):
        return True
    if "<tool_call>\n</think>" in messages[-1]["content"]:
        return True
    num_pairs = re.findall(r"<tool_call>.*?</tool_call>", messages[-1]["content"], re.DOTALL)
    num_open = re.findall(r"<tool_call>", messages[-1]["content"], re.DOTALL)
    num_close = re.findall(r"</tool_call>", messages[-1]["content"], re.DOTALL)
    if len(num_pairs) != len(num_open) or len(num_pairs) != len(num_close):
        return True
    return False


def check_data(data_dir):
    # filenames = [d for d in os.listdir(data_dir) if d != "train.jsonl"]
    filenames = ["train.jsonl"]
    dirty_data = {}
    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        with open(file_path) as f:
            for line in f:
                d = json.loads(line)
                if is_dirty(d["messages"]):
                    dirty_data.setdefault(filename, []).append(d)
                    print("=" * 40)
                    print(f"Found dirty data in {filename}: {d['messages'][-1]['content']}")
                    print("=" * 40)
    if dirty_data:
        print("Dirty data found:")
        for filename, entries in dirty_data.items():
            print(f"{filename}: {len(entries)} entries")
    else:
        print("What a nice clean dataset!")


if __name__ == "__main__":
    check_data("data/agent-26")
