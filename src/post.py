import requests
import yaml


DEBUG = True

def call_llm(url, model, prompt, system_prompt=None, **sampling_params):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        **sampling_params,
    }
    if DEBUG:
        for msg in messages:
            print("=" * 20)
            print(f"{msg['role']}: {msg['content']}")
        print("=" * 20)
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


def condense(url, model, **sampling_params):
    history = []
    while True:
        with open("template/prompt.yaml") as f:
            templates = yaml.safe_load(f)
        template = templates["condense"]
        new_question = input("[User]: ")
        
        if new_question.lower() == "exit":
            break
        if new_question.lower() == "clear":
            history = []
            print("[Info]: History cleared.")
            continue
            
        if len(history) > 0:
            prompt = template.format(history="\n".join(history), new_question=new_question)
            # prompt += "\n\\no_think"
            try:
                answer = call_llm(url, model, prompt, **sampling_params)
                print(f"[Condense]: {answer}")
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                continue
        history.append(f"<question>\n{new_question}\n</question>")


if __name__ == "__main__":
    url = "http://127.0.0.1:5090/v1/chat/completions"
    model = "qwen3_1.7b"
    # print(call_llm(url, model, "你是谁\\no_think"))
    s = """你好，你是谁 /no_think"""
    resp = call_llm(url, model, s, temperature=0)
    print(resp)
    # condense(url, model, temperature=0)
