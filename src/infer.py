import pandas as pd
import os
import yaml
devices = [3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])

from vllm import LLM, SamplingParams


data_files = [
    {"path": "data/raw/groupby.xlsx", "template": "group_by", "variables": ["Input", "env"]}
]
index = 0
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
model_path = "ft-models/v1-20250507-102226/checkpoint-328"

with open("template/prompt.yaml") as f:
    templates = yaml.safe_load(f)
template = templates[data_files[index]["template"]].strip()

data = []
df = pd.read_excel(data_files[index]["path"])
for _, row in df.iterrows():
    prompt = template.format(**{var: row[var] for var in data_files[index]["variables"]})
    # print(prompt)
    data.append([{"role": "user", "content": prompt}])

llm = LLM(model=model_path, trust_remote_code=True, max_seq_len_to_capture=2048, tensor_parallel_size=len(devices))
outputs = llm.chat(data, sampling_params)

for i, output in enumerate(outputs):
    result = output.outputs[0].text.strip()
    if "</think>" in result:
        result = result.split("</think>")[-1]
    df.loc[i, "Output"] = result

df.to_excel(data_files[index]["path"], index=False)