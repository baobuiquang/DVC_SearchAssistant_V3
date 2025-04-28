from pkg.HYSE.HYSE import HYSE_EngineHybrid, json2dict, dict2json
from pkg.NLPT.NLPT import NLPT_Normalize
from pkg.LLM.LLM import Process_LLM

import numpy as np
import json
import re

def print_dict(d, indent=4):
    if not isinstance(d, dict):
        print("Input is not a dictionary.")
        return
    pretty_string = json.dumps(d, indent=indent, ensure_ascii=False)
    print(pretty_string)

def print_list(l):
    for e in l:
        print(e)

def create_prompt_1(inputtext, hyse_res):
    prompt_list_1 = [{"Tên thủ tục": e['doc'], "INDEX": str(e['index'])} for e in hyse_res]
    prompt_schema_1 = """\
{
    "type": "object",
    "properties": {
        "Tên thủ tục": {"type": "string", "description": "Tên của thủ tục liên quan nhất"}
        "INDEX": {"type": "string", "description": "INDEX của thủ tục liên quan nhất"},
    }
}"""
    prompt_1 = f"""\
Bạn sẽ được cung cấp: (1) Câu hỏi của người dùng, (2) Danh sách thủ tục hiện có, và (3) Schema cấu trúc của kết quả.
Nhiệm vụ của bạn là: (4) Trích xuất duy nhất 1 thủ tục liên quan nhất đến câu hỏi của người dùng.

### (1) Câu hỏi của người dùng:
"{inputtext}"

### (2) Danh sách thủ tục hiện có:
{prompt_list_1}

### (3) Schema cấu trúc của kết quả:
{prompt_schema_1}

### (4) Nhiệm vụ:
Từ câu hỏi của người dùng "{inputtext}", tìm ra duy nhất 1 thủ tục liên quan nhất đến câu hỏi, tuân thủ schema một cách chính xác.
Lưu ý quan trọng: Nếu không có thủ tục nào liên quan, trả về "Không có thủ tục liên quan".
Định dạng kết quả: Không giải thích, không bình luận, không văn bản thừa. Chỉ trả về kết quả JSON hợp lệ. Bắt đầu bằng "{{", kết thúc bằng "}}".
"""
    return prompt_1

def preprocess_fieldtext_html_in_DATA(DATA):
    def preprocess_fieldtext_html(fld_text):
        fld_text = fld_text.replace("<br/>", "")                             # Pre-processing: Remove all <br/>
        fld_text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'\1', fld_text)   # Pre-processing: Remove <strong> tag (keep content)
        fld_text = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', fld_text)       # Pre-processing: Remove <span> tag (keep content)
        fld_text = re.sub(r'<em[^>]*>(.*?)</em>', r'\1', fld_text)           # Pre-processing: Remove <em> tag (keep content)
        fld_text = re.sub(r'<i[^>]*>(.*?)</i>', r'\1', fld_text)             # Pre-processing: Remove <i> tag (keep content)
        fld_text = re.sub(r'<b[^>]*>(.*?)</b>', r'\1', fld_text)             # Pre-processing: Remove <b> tag (keep content)
        fld_text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', fld_text)             # Pre-processing: Remove <a> tag (keep content)
        fld_text = re.sub(r'<u[^>]*>(.*?)</u>', r'\1', fld_text)             # Pre-processing: Remove <u> tag (keep content)
        fld_text = re.sub(r"\s*class=['\"][^'\"]*['\"]", '', fld_text)       # Pre-processing: Remove class="something"
        fld_text = re.sub(r"\s*style=['\"][^'\"]*['\"]", '', fld_text)       # Pre-processing: Remove style="something"
        return fld_text
    for infopool_id in list(DATA.keys()):
        for iii, thutuc in enumerate(DATA[infopool_id]["data"]):
            fieldnames = list(thutuc["content"].keys())
            for fld in fieldnames:
                DATA[infopool_id]["data"][iii]["content"][fld] = preprocess_fieldtext_html(thutuc["content"][fld])
    return DATA

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

# Read DATA
DATA = json2dict("static/DATA.json")
DATA = preprocess_fieldtext_html_in_DATA(DATA)

INFOPOOL_DATAS = {}
INFOPOOL_HYSE_ENGINES = {}
for infopool_id in list(DATA.keys()):
    INFOPOOL_DATAS[infopool_id] = DATA[infopool_id]["data"]
    INFOPOOL_HYSE_ENGINES[infopool_id] = HYSE_EngineHybrid(name=infopool_id)
    INFOPOOL_HYSE_ENGINES[infopool_id].update([e["name"] for e in DATA[infopool_id]["data"]])

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

def find_bestthutuc_and_suggestions(inputtext, infopool_id):
    CURRENT_INFOPOOL_ID = infopool_id
    hyse_res = INFOPOOL_HYSE_ENGINES[CURRENT_INFOPOOL_ID].search([inputtext])[0]
    bestthutuc = {}
    suggestions = []
    for _ in range(3):
        llmres1 = Process_LLM(create_prompt_1(inputtext, hyse_res))
        regexmatch1 = re.search(r'\{.*\}', llmres1, re.S)
        if regexmatch1:
            try:
                jsonobj1 = json.loads(regexmatch1.group())
                bestthutuc = INFOPOOL_DATAS[CURRENT_INFOPOOL_ID][int(jsonobj1["INDEX"])]
                if bestthutuc["name"] == jsonobj1["Tên thủ tục"]:
                    # -------------------------------------------------- # bestthutuc
                    # print_dict(bestthutuc)
                    # -------------------------------------------------- # suggestions
                    MIN_SIM_TO_BE_SUGGESTED = 0.92
                    hyse_res_embs = np.array([INFOPOOL_HYSE_ENGINES[CURRENT_INFOPOOL_ID].search_engine_3.embs[e2] for e2 in [e1["index"] for e1 in hyse_res]])
                    similarities = hyse_res_embs @ hyse_res_embs.T
                    bestthutuc_pos = [e3["doc"] for e3 in hyse_res].index(bestthutuc["name"])
                    suggestions = [INFOPOOL_DATAS[CURRENT_INFOPOOL_ID][hyse_res[ii]["index"]] for ii, sim in sorted(enumerate(similarities[bestthutuc_pos]), key=lambda x: -x[1]) if ii != bestthutuc_pos and sim > MIN_SIM_TO_BE_SUGGESTED]
                    suggestions = [{"name": e4["name"], "link": e4["link"], "code": e4["code"]} for e4 in suggestions]
                    # print_list(suggestions)
                    # -------------------------------------------------- # 
                    break
            except:
                pass
    return bestthutuc, suggestions

def create_api_content_data(bestthutuc):
    fieldnames1 = list(bestthutuc["content"].keys())
    fieldnames2 = [NLPT_Normalize(e, lower=True, remove_diacritics=True, replace_spacelikes_with_1space=True, remove_punctuations=True).replace(" ", "_") for e in fieldnames1]
    content_data = {}
    for fld1, fld2 in zip(fieldnames1, fieldnames2):
        content_data[fld2] = bestthutuc["content"][fld1]
    return content_data

def create_api_content_0(bestthutuc):
    MAXWORDS = 100
    XEMCHITIET = f"""... <a href='{bestthutuc['link']}' target='_blank'>(xem chi tiết ↗)</a>"""
    fieldnames = list(bestthutuc["content"].keys())
    content_0 = ""
    content_0 += f"""<a href='{bestthutuc['link']}' target='_blank'><h2>Thủ tục: {bestthutuc['name']}</h2></a>"""
    for fld in fieldnames:
        # ----------
        bestthutuc_content_fld = bestthutuc["content"][fld]
        # ----------
        content_0 += f"""<h3>{fld}</h3>"""
        content_0 += "<p>"
        if len(bestthutuc_content_fld.split(" ")) < MAXWORDS:
            content_0 += f"""{bestthutuc_content_fld}"""
        else:
            content_0 += f"""{" ".join(bestthutuc_content_fld.split(" ")[:MAXWORDS])}"""
            # ----- # Gradio fix for trimmed text
            if "<ul>" in bestthutuc_content_fld:    content_0 += "</ul>"
            if "<ol>" in bestthutuc_content_fld:    content_0 += "</ol>"
            if "<table>" in bestthutuc_content_fld: content_0 += "</td></tr></tbody></table>"
            # -----
            content_0 += XEMCHITIET
        content_0 += "</p>"
        # -----
    content_0 += f"""<h3>Xem đầy đủ văn bản thủ tục tại:</h3>"""
    content_0 += f"""<a href='{bestthutuc['link']}' target='_blank'>{bestthutuc['link']}</a>"""
    return content_0

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

def DVC_SearchAssistant(inputtext, infopool_id):
    try:
        bestthutuc, suggestions = find_bestthutuc_and_suggestions(inputtext, infopool_id)
        API_OBJECT = {
            "input": inputtext,
            "datapool": infopool_id,
            "name": bestthutuc["name"],
            "link": bestthutuc["link"],
            "code": bestthutuc["code"],
            "content_0": create_api_content_0(bestthutuc),
            "content_1": "DO-NOT-USE-THIS",
            "content_2": "DO-NOT-USE-THIS",
            "content_data": create_api_content_data(bestthutuc),
            "suggestions": suggestions,
        }
        return API_OBJECT
    except:
        API_OBJECT = {
            "input": inputtext,
            "datapool": infopool_id,
            "name": "",
            "link": "",
            "code": "",
            "content_0": "Xin chào, mình có thể giúp gì cho bạn?",
            "content_1": "DO-NOT-USE-THIS",
            "content_2": "DO-NOT-USE-THIS",
            "content_data": {},
            "suggestions": [],
        }
        return API_OBJECT


# inputtext = "tôi muốn cưới chồng người nước ngoài"
# infopool_id = "DVC_TTHC_LamDong"
# # infopool_id = "DVC_TTHC_LangSon"
# DVC_SearchAssistant(inputtext, infopool_id)