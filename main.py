from DVC_SearchAssistant import DVC_SearchAssistant

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# ==================================================================================================== 🏓 Ping to server to confirm alive
# ====================================================================================================
# ====================================================================================================

from contextlib import asynccontextmanager
import requests
import asyncio
async def ping_to_server_to_confirm_alive():
    while True:
        try:
            r = requests.post("https://pims.lamdongtructuyen.vn/hub/api/ServiceHistory/SaveLog", json={"LastCode":200,"Url":"","Token":"OKFGFwXXnCYu9P5zpA3iX4v6roKEh6GU"})
            print(f"> 🏓 Ping to server to confirm alive! - DVC_TTHC_LamDong: {r.text}")
            r = requests.post("https://pims.lamdongtructuyen.vn/hub/api/ServiceHistory/SaveLog", json={"LastCode":200,"Url":"","Token":"uIMtpx3TjutUijfEanqh4s1C9Nhj5cSI"})
            print(f"> 🏓 Ping to server to confirm alive! - DVC_TTHC_LangSon: {r.text}")
        except Exception as er:
            print(f"⚠️ DVC_SearchAssistant > 🏓 > Error: {er}")
        await asyncio.sleep(600)  # in seconds (600 seconds = 10 minutes)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    task = asyncio.create_task(ping_to_server_to_confirm_alive())
    yield
    # Shutdown (optional: cancel background task)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("⚠️ DVC_SearchAssistant > 🏓 > asyncio.CancelledError")

# ==================================================================================================== ⚡ API
# ====================================================================================================
# ====================================================================================================

app = FastAPI(title="DVC_SearchAssistant_V3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GLOBAL_AUTHENKEY = "VNPT"

class SearchAssistantPlayload(BaseModel):
    authen_key : str = "VNPT"
    datapool: str = "DVC_TTHC_LamDong"
    input: str = "Tôi muốn thành lập công ty thì cần phải làm gì?"

# ENDPOINT: /
@app.get("/", include_in_schema=False)
async def endpoint_root():
    return {"message": "API is running..."}

# ENDPOINT: /api
@app.post("/api", tags=["Main API Endpoint"])
async def endpoint_api(playload:SearchAssistantPlayload):
    if GLOBAL_AUTHENKEY != playload.authen_key:
        return {"message": "🧱 Your request is not authenticated!"}
    else:
        try:
            # --------------------------------------------------
            return DVC_SearchAssistant(inputtext=playload.input, infopool_id=playload.datapool) # ✨
            # --------------------------------------------------
        except Exception as e: return {"message": f"⚠️ Error: {e}"}

# ENDPOINT: /KILL
@app.get("/kill", include_in_schema=False)
async def Endpoint_Kill():
    import os
    import signal
    os.kill(os.getpid(), signal.SIGTERM)

# ==================================================================================================== 📱 DEMO APP
# ====================================================================================================
# ====================================================================================================

import gradio as gr
import time
import ast
def str2whatitpresent(s):
    return ast.literal_eval(s)

theme = gr.themes.Base(
    # primary_hue="neutral",
    # secondary_hue="neutral",
    # neutral_hue="neutral",
    font=[gr.themes.GoogleFont('Inter')], 
    font_mono=[gr.themes.GoogleFont('Ubuntu Mono')]
)
head = """
<link rel="icon" href="https://raw.githubusercontent.com/baobuiquang/DVC_SearchAssistant_V3/refs/heads/main/static/logo.png">
"""
css = """
* { -ms-overflow-style: none; scrollbar-width: none; }
*::-webkit-scrollbar { display: none; }
footer { display: none !important; }
main { padding: 0 !important; }
#gr_row { gap: 0; }
#gr_variable_1, #gr_var_prompt { display: none !important; }
button.upload-button { display: none !important; }
#gr_mid_column { height: 98svh; }
#gr_history { flex-grow: 1; }
.icon-button-wrapper button[title="Clear"]::after { content: "Clear"; padding: 0 2px; }
.message.bot { margin-top: 16px !important; }
.message-content { margin: 16px 8px !important; }
textarea { font-size: 16px !important; }
#gr_textbox { border: 2px solid hsl(0 0 90) !important; }
#gr_footnote * { color: hsl(0 0 70); font-size: 12px; font-style: italic; text-align: center; }
#gr_infopool { padding: 16px; }
#gr_infopool * { color: hsl(0 0 20); }
#gr_history .placeholder img { height: 32px; }
button.example { padding: 10px 14px !important; }
button.example .example-content { justify-content: center !important; }
button.example .example-text-content { margin-top: 0 !important; }
button.example span.example-text { font-size: 13px !important; }
"""

def fn_select_example(evt: gr.SelectData):
    return {'text': evt.value['text'], 'files': [e['path'] for e in evt.value['files']]}

def fn_chat_1(gr_history, gr_textbox):
    for filepath in gr_textbox["files"]:
        gr_history += [{"role": "user", "content": gr.File(filepath)}]
    gr_history += [{"role": "user", "content": gr_textbox["text"]}]
    # 
    gr_variable_1 = str(gr_textbox)
    # 
    gr_history_clean = [{"role": e["role"], "content": e["content"]} for e in gr_history if isinstance(e["content"], str)]
    gr_var_prompt = str(gr_history_clean)
    # 
    return gr_history, "", gr_variable_1, gr_var_prompt

def fn_chat_2(gr_var_prompt, gr_history, gr_infopool):
    # --------------------------------------------------
    api_res = DVC_SearchAssistant(inputtext=str2whatitpresent(gr_var_prompt)[-1]['content'], infopool_id=gr_infopool.strip()) # ✨
    # --------------------------------------------------
    bot_responses = []
    if len(api_res["suggestions"]) > 0:
        suggestions_text = "## Gợi ý liên quan:\n"
        for sugg in api_res["suggestions"]: suggestions_text += f"* `{sugg['code']}` [{sugg['name']}]({sugg['link']})\n"
        bot_responses = [api_res["content_0"], suggestions_text]
    else:
        bot_responses = [api_res["content_0"]]
    # --------------------------------------------------
    for bot_response in [str(ee) for ee in bot_responses]:
        gr_history += [{"role": "assistant", "content": ""}]
        for token in bot_response.split(" "):
            gr_history[-1]["content"] += token + " "
            time.sleep(0.01)
            yield gr_history

with gr.Blocks(title="Demo - DVC_SearchAssistant_V3", theme=theme, css=css, head=head, analytics_enabled=False, fill_height=True, fill_width=True) as demo:
    with gr.Row(elem_id="gr_row"):
        with gr.Column(min_width=1):
            gr_variable_1 = gr.Code(interactive=False, wrap_lines=True, label="gr_variable_1", elem_id="gr_variable_1")
            gr_var_prompt = gr.Code(interactive=False, wrap_lines=True, label="gr_var_prompt", elem_id="gr_var_prompt")
            gr_infopool = gr.Radio(choices=["DVC_TTHC_BoCongAn", "DVC_TTHC_LamDong", "DVC_TTHC_LangSon"], elem_id="gr_infopool", value="DVC_TTHC_LamDong", type="value", info="### Nguồn dữ liệu", interactive=True, container=False)
        with gr.Column(elem_id="gr_mid_column", min_width=800):
            gr_history = gr.Chatbot(elem_id="gr_history", type="messages", placeholder="![image](https://raw.githubusercontent.com/baobuiquang/DVC_SearchAssistant_V2/refs/heads/main/static/logo.png)\n## Xin chào!\nMình là chatbot hỗ trợ tìm kiếm thủ tục dịch vụ công.", avatar_images=(None, "https://raw.githubusercontent.com/baobuiquang/DVC_SearchAssistant_V3/refs/heads/main/static/logo.png"), group_consecutive_messages=False, container=False, examples=[
                {"text": "Hướng dẫn sử dụng", "files": []},
                {"text": "cach nop ho so", "files": []},
                {"text": "Hướng dẫn làm thủ tục", "files": []},
                {"text": "Tra cứu hồ sơ", "files": []},
                {"text": "sđt đường dây nóng hỗ trợ", "files": []},
                {"text": "Điều kiện đăng ký kết hôn là gì?", "files": []},
                {"text": "Thành phần hồ sơ thành lập doanh nghiệp tư nhân", "files": []},
                {"text": "Thời gian giải quyết phúc khảo bài thi tốt nghiệp là bao lâu?", "files": []},
                {"text": "can cu phap ly dang ky khai sinh", "files": []},
                {"text": "Lệ phí thành lập hộ kinh doanh là bao nhiêu?", "files": []},
                {"text": "Thành phần hồ sơ cấp lý lịch tư pháp gồm những gì?", "files": []},
                {"text": "Thành lập doanh nghiệp tư nhân yêu cầu những gì?", "files": []},
                {"text": "Vợ tôi sắp sinh con tôi cần làm gì?", "files": []},
                {"text": "Giấy tờ cần thiết để mình khởi nghiệp.", "files": []},
                {"text": "Tôi muốn tố cáo hàng xóm trồng cần sa.", "files": []},
                {"text": "Tôi muốn thành lập công ty tnhh 1 thành viên", "files": []},
                {"text": "Tôi muốn thành lập công ty tnhh 2 thành viên", "files": []},
                {"text": "Tôi muốn thành lập công ty tnhh 9 thành viên", "files": []},
                {"text": "Đấu thầu đất xây dựng", "files": []},
                {"text": "Làm sao để cưới chồng?", "files": []},
                {"text": "Đất đai", "files": []},
                {"text": "Thủ tục chuyển trường cấp 3", "files": []},
                {"text": "Cấp lý lịch tư pháp", "files": []},
                {"text": "Cháu muốn phúc khảo bài thi thpt", "files": []},
            ])
            gr_textbox = gr.MultimodalTextbox(elem_id="gr_textbox", file_count="multiple", placeholder="Nhập câu hỏi ở đây", submit_btn=True, autofocus=True, autoscroll=True, container=False)
            gr.Markdown("DVC_SearchAssistant_V3", elem_id="gr_footnote", container=False)
        with gr.Column(min_width=1):
            gr.Markdown("")
    gr.on(triggers=gr_textbox.submit, fn=fn_chat_1, inputs=[gr_history, gr_textbox], outputs=[gr_history, gr_textbox, gr_variable_1, gr_var_prompt], show_progress="hidden"
    ).then(fn=fn_chat_2, inputs=[gr_var_prompt, gr_history, gr_infopool], outputs=[gr_history], show_progress="full")
    gr.on(fn=fn_select_example, triggers=[gr_history.example_select], outputs=[gr_textbox], show_progress="hidden")
app = gr.mount_gradio_app(app, demo, path="/demo")

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

if __name__ == "__main__":
    print("> Docs: http://127.0.0.1:5005/docs")
    print("> Demo: http://127.0.0.1:5005/demo?__theme=light")
    uvicorn.run(app, host = "0.0.0.0", port = 5005)