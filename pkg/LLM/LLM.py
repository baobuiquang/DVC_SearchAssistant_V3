# def activate_pkg():
#     import os
#     import sys
#     p = os.getcwd()
#     while not os.path.isdir(os.path.join(p, "pkg")): p = os.path.dirname(p)
#     sys.path.append(p) if p not in sys.path else None
# activate_pkg()

DEFAULT_VENDOR = "vnpt"

import requests
import json
import ast
def str2whatitpresent(s):
    return ast.literal_eval(s)

KEY_OPENROUTER = "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"
KEY_DEEPINFRA  = "rD6SaD8vai3LxvehBVTyvq2djEjCIzGQ"
KEY_OLLAMA     = "ollama"
KEY_VNPT       = "ollama"
URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
URL_DEEPINFRA  = "https://api.deepinfra.com/v1/openai/chat/completions"
URL_OLLAMA     = "http://localhost:11434/v1/chat/completions"
URL_VNPT       = "http://192.168.80.99:11434/v1/chat/completions"
MDL_OPENROUTER = "qwen/qwen-2.5-7b-instruct"
MDL_DEEPINFRA  = "Qwen/Qwen2.5-7B-Instruct"
MDL_OLLAMA     = "qwen2.5:7b-instruct-q4_K_M"
# 🍌👇
# MDL_VNPT       = "gemma3:12b"
# MDL_VNPT       = "gemma3:12b-it-qat"
MDL_VNPT       = "mygemma:12b"

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

class RequestInput:
    def __init__(self, prompt, stream=False, vendor=DEFAULT_VENDOR):
        if vendor=="ollama":
            LLM_API_KEY = KEY_OLLAMA
            LLM_API_URL = URL_OLLAMA
            LLM_API_MDL = MDL_OLLAMA
        elif vendor=="openrouter":
            LLM_API_KEY = KEY_OPENROUTER
            LLM_API_URL = URL_OPENROUTER
            LLM_API_MDL = MDL_OPENROUTER
        elif vendor=="deepinfra":
            LLM_API_KEY = KEY_DEEPINFRA
            LLM_API_URL = URL_DEEPINFRA
            LLM_API_MDL = MDL_DEEPINFRA
        elif vendor=="vnpt":
            LLM_API_KEY = KEY_VNPT
            LLM_API_URL = URL_VNPT
            LLM_API_MDL = MDL_VNPT
        self.url = LLM_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        self.stream = stream
        if isinstance(prompt, str):                                       # Case 1: prompt is a string
            if prompt[0] != "[" or prompt[-1] != "]":                     # Case 1a: prompt is a single question (a string)
                self.payload = {
                    "stream": stream,
                    "model": LLM_API_MDL,
                    "messages": [ { "role": "user", "content": prompt } ]
                }
            else:                                                         # Case 1b: prompt is a history (a string but represents a list)
                self.payload = {
                    "stream": stream,
                    "model": LLM_API_MDL,
                    "messages": str2whatitpresent(prompt)
                }
        elif isinstance(prompt, list):                                    # Case 2: prompt is a history (a list)
            self.payload = {
                "stream": stream,
                "model": LLM_API_MDL,
                "messages": prompt
            }
        else:
            raise ValueError("⚠️ LLM > RequestInput > prompt is neither not string nor list")

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

# Non-streaming
def Process_LLM(prompt):
    try:
        reqin = RequestInput(prompt=prompt, stream=False)
        with requests.post(url=reqin.url, headers=reqin.headers, json=reqin.payload, stream=reqin.stream) as req:
            return req.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ LLM > {e}")
        return "⚠️"

# Streaming
def Process_LLM_Stream(prompt, history=[]):
    history += [{"role":"assistant", "content":""}]
    try:
        reqin = RequestInput(prompt=prompt, stream=True)
        with requests.post(url=reqin.url, headers=reqin.headers, json=reqin.payload, stream=reqin.stream) as req:
            for chunk in req.iter_lines():
                if chunk:
                    chunk = chunk.decode("utf-8", errors="replace")[6:]
                    try:
                        chunk = json.loads(chunk)["choices"][0]["delta"]["content"]
                        history[-1]["content"] += chunk
                        yield history
                    except:
                        pass
    except Exception as e:
        print(f"⚠️ LLM > {e}")
        history[-1]["content"] += "⚠️"
        yield history

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

# from pkg.LLM.LLM import Process_LLM
# Process_LLM("Tên bạn là gì?")

# prompt = "1 + 2 bằng bao nhiêu? Hãy giải thích."
# prompt = [ { "role": "user", "content": "1 + 2 bằng bao nhiêu? Hãy giải thích." } ]
# Process_LLM(prompt)
# for chunk in Process_LLM_Stream(prompt):
#     print(chunk)