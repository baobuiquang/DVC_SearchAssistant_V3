DEFAULT_VENDOR = "vnpt"
KEY_VNPT       = "ollama"
URL_VNPT       = "http://192.168.20.62:11434/v1/chat/completions"
MDL_VNPT       = "gemma3:4b"

import requests
import json
import ast
def str2whatitpresent(s):
    return ast.literal_eval(s)

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

class RequestInput:
    def __init__(self, prompt, stream=False, vendor=DEFAULT_VENDOR):
        if vendor=="vnpt":
            LLM_API_KEY = KEY_VNPT
            LLM_API_URL = URL_VNPT
            LLM_API_MDL = MDL_VNPT
        self.url = LLM_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        self.stream = stream
        if isinstance(prompt, str):                                       # 
            if prompt[0] != "[" or prompt[-1] != "]":                     # 
                self.payload = {
                    "stream": stream,
                    "model": LLM_API_MDL,
                    "messages": [ { "role": "user", "content": prompt } ]
                }
            else:                                                         # 
                self.payload = {
                    "stream": stream,
                    "model": LLM_API_MDL,
                    "messages": str2whatitpresent(prompt)
                }
        elif isinstance(prompt, list):                                    # 
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