from pkg.HYSE.HYSE import HYSE_EngineHybrid, json2dict, dict2json
from pkg.NLPT.NLPT import NLPT_Normalize
from pkg.LLM.LLM import Process_LLM

from static.DATA_HARDPARAPHRASE import DATA_HARDPARAPHRASE
from static.DATA_HARDCODE import DATA_HARDCODE

import numpy as np
import logging
import json
import re
import os

os.makedirs("_log", exist_ok=True)
file_handler = logging.FileHandler('_log/log.txt', encoding='utf-8')
logging.basicConfig(
    handlers=[file_handler],
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%y%m%d%H%M%S'
)

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

def print_dict(d, indent=4):
    if not isinstance(d, dict):
        print("Input is not a dictionary.")
        return
    pretty_string = json.dumps(d, indent=indent, ensure_ascii=False)
    print(pretty_string)

def print_list(l):
    for e in l:
        print(e)

# CONVERT: ['hướng dẫn', 'hồ sơ']
# TO:      ['hướng dẫn', 'hướng dan', 'huong dẫn', 'huong dan', 'hồ sơ', 'hồ so', 'ho sơ', 'ho so']
def create_normalied_list_of_text(myls):
    def combine_lists_with_spaces(list1, list2):
        result = []
        def backtrack(index=0, current=""):
            if index == len(list1):
                result.append(current.strip()) 
                return
            sep = " " if index < len(list1) - 1 else ""
            backtrack(index + 1, current + list1[index] + sep)
            backtrack(index + 1, current + list2[index] + sep)
        backtrack()
        return result
    res = [item for sublist in [combine_lists_with_spaces(ele.split(), [NLPT_Normalize(el, lower=True, remove_diacritics=True, replace_spacelikes_with_1space=True, remove_punctuations=True) for el in ele.split()]) for ele in myls] for item in sublist]
    return res

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
        fld_text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\1', fld_text)           # Pre-processing: Remove <h1> tag (keep content)
        fld_text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\1', fld_text)           # Pre-processing: Remove <h2> tag (keep content)
        fld_text = re.sub(r'<h3[^>]*>(.*?)</h4>', r'\1', fld_text)           # Pre-processing: Remove <h3> tag (keep content)
        fld_text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\1', fld_text)           # Pre-processing: Remove <h4> tag (keep content)
        fld_text = re.sub(r'<h5[^>]*>(.*?)</h5>', r'\1', fld_text)           # Pre-processing: Remove <h5> tag (keep content)
        fld_text = re.sub(r'<h6[^>]*>(.*?)</h6>', r'\1', fld_text)           # Pre-processing: Remove <h6> tag (keep content)
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

def create_api_content_0(inputtext, bestthutuc):
    OPINIONATED_FIELD_TRIGGERS = [
        { "fieldnames": ['Thành phần hồ sơ'], "triggers": ['thành phần hồ sơ', 'thành phần', 'hồ sơ'] },
        { "fieldnames": ['Cách thức thực hiện'], "triggers": ['cách thức thực hiện', 'cách thức', 'thực hiện'] },
        { "fieldnames": ['Trình tự thực hiện'], "triggers": ['trình tự thực hiện', 'trình tự', 'thực hiện'] },
        { "fieldnames": ['Thời gian giải quyết', 'Thời hạn giải quyết'], "triggers": ['thời gian giải quyết', 'thời hạn giải quyết', 'thời gian', 'thời hạn'] },
        { "fieldnames": ['Yêu cầu - điều kiện', 'Yêu cầu, điều kiện'], "triggers": ['yêu cầu - điều kiện', 'yêu cầu, điều kiện', 'yêu cầu', 'điều kiện'] },
        { "fieldnames": ['Đối tượng thực hiện'], "triggers": ['đối tượng thực hiện', 'đối tượng', 'thực hiện'] },
        { "fieldnames": ['Căn cứ pháp lý'], "triggers": ['căn cứ pháp lý', 'căn cứ', 'pháp lý'] },
        { "fieldnames": ['Biểu mẫu đính kèm', 'Tên mẫu đơn, tờ khai'], "triggers": ['biểu mẫu đính kèm', 'tên mẫu đơn, tờ khai', 'biểu mẫu', 'mẫu đơn', 'tờ khai'] },
        { "fieldnames": ['Phí, lệ phí', 'Lệ Phí', 'Phí'], "triggers": ['phí, lệ phí', 'lệ phí', 'chi phí', 'bao nhiêu tiền'] },
        { "fieldnames": ['Lĩnh vực'], "triggers": ['lĩnh vực'] },
        { "fieldnames": ['Cơ quan thực hiện'], "triggers": ['cơ quan thực hiện', 'thực hiện'] },
        { "fieldnames": ['Kết quả thực hiện', 'Kết quả'], "triggers": ['kết quả thực hiện', 'kết quả', 'thực hiện'] },
        { "fieldnames": ['Địa chỉ tiếp nhận'], "triggers": ['địa chỉ tiếp nhận', 'nơi tiếp nhận', 'tiếp nhận'] },
        { "fieldnames": ['Số lượng bộ hồ sơ'], "triggers": ['số lượng bộ hồ sơ', 'số lượng', 'hồ sơ'] },
    ]
    flag_there_is_trigger = False
    TRIGGERED_FIELDS = []
    for ee in OPINIONATED_FIELD_TRIGGERS:
        for trigger in create_normalied_list_of_text(ee["triggers"]):
            if trigger.lower() in inputtext.lower():
                for fld in ee["fieldnames"]:
                    if fld in list(bestthutuc["content"].keys()):
                        TRIGGERED_FIELDS.append(fld)
                flag_there_is_trigger = True
                break
    # ----------------------------------------------------------------------------------------------------
    if flag_there_is_trigger:
        MAXWORDS = 9999
        SELECTED_FIELDS = TRIGGERED_FIELDS
    else:
        MAXWORDS = 100
        SELECTED_FIELDS = [
            "Thành phần hồ sơ", 
            "Trình tự thực hiện", 
            "Cách thức thực hiện", 
            "Thời gian giải quyết", "Thời hạn giải quyết",
            "Phí, lệ phí", "Phí", "Lệ Phí",
            # "Kết quả", "Kết quả thực hiện",
        ]
    XEMCHITIET = f"""... <a href='{bestthutuc['link']}' target='_blank'>(xem chi tiết ↗)</a>"""
    content_0 = ""
    content_0 += f"""<a href='{bestthutuc['link']}' target='_blank'><h2>Thủ tục: {bestthutuc['name']}</h2></a>"""
    for fld in SELECTED_FIELDS:
        if fld in list(bestthutuc["content"].keys()):
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
    content_0 += f"""<h3>Xem văn bản đầy đủ tại:</h3>"""
    content_0 += f"""<a href='{bestthutuc['link']}' target='_blank'>{bestthutuc['link']}</a>"""
    return content_0

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

def DVC_SearchAssistant(inputtext, infopool_id):
    # ----- Log 📄
    logging.info(f"infopool_id='{infopool_id}' - inputtext='{inputtext}'")
    # -----
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
    # --------------------------------------------------
    def inputtext_preprocessing(inputtext):
        return NLPT_Normalize(inputtext, replace_spacelikes_with_1space=True) # 🍌 Opinionated inputtext pre-processing
    inputtext = inputtext_preprocessing(inputtext)
    # -------------------------------------------------- 1️⃣ Special Case 1: inputtext is empty
    if inputtext == "":
        return API_OBJECT
    # -------------------------------------------------- 2️⃣ Special Case 2A: inputtext contains DATA_HARDPARAPHRASE
    for hardparaphrase in DATA_HARDPARAPHRASE:
        possible_keywords = create_normalied_list_of_text(hardparaphrase["keywords"])
        for k in possible_keywords:
            if k.lower() in inputtext.lower():
                inputtext += " " + hardparaphrase["append_phrase"]
                print(f"> inputtext: {inputtext}")
                break
    # -------------------------------------------------- 2️⃣ Special Case 2B: inputtext is DATA_HARDCODE
    for faq in DATA_HARDCODE[infopool_id]:
        possible_faq_questions = create_normalied_list_of_text(faq["questions"])
        for e in possible_faq_questions:
            if e.lower() in inputtext.lower():
                API_OBJECT = faq["answer"]
                API_OBJECT["input"] = inputtext
                return API_OBJECT
    # -------------------------------------------------- 3️⃣ Case 3: inputtext is normal search text -> LLM
    try:
        bestthutuc, suggestions = find_bestthutuc_and_suggestions(inputtext, infopool_id)
        API_OBJECT["name"] = bestthutuc["name"]
        API_OBJECT["link"] = bestthutuc["link"]
        API_OBJECT["code"] = bestthutuc["code"]
        API_OBJECT["content_0"] = create_api_content_0(inputtext, bestthutuc)
        API_OBJECT["content_1"] = "DO-NOT-USE-THIS"
        API_OBJECT["content_2"] = "DO-NOT-USE-THIS"
        API_OBJECT["content_data"] = create_api_content_data(bestthutuc)
        API_OBJECT["suggestions"] = suggestions
        return API_OBJECT
    except Exception as er:
        print(f"⚠️ DVC_SearchAssistant > Error: {er}")
    # --------------------------------------------------
    return API_OBJECT


# inputtext = "tôi muốn cưới chồng người nước ngoài"
# infopool_id = "DVC_TTHC_LamDong"
# # infopool_id = "DVC_TTHC_LangSon"
# DVC_SearchAssistant(inputtext, infopool_id)