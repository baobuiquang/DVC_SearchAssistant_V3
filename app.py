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

    bot_responses = [
        f"Your message: {str2whatitpresent(gr_var_prompt)[-1]['content']}",
        f"Datapool: {gr_infopool}",
        "Hello 1 - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Hello 2 - Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Hello 3 - Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    ]
    for bot_response in bot_responses:
        gr_history += [{"role": "assistant", "content": ""}]
        for token in bot_response.split(" "):
            gr_history[-1]["content"] += token + " "
            time.sleep(0.08)
            yield gr_history

# ====================================================================================================

with gr.Blocks(title="DVC_SearchAssistant_V3", theme=theme, css=css, head=head, analytics_enabled=False, fill_height=True, fill_width=True) as demo:
    with gr.Row(elem_id="gr_row"):
        with gr.Column(min_width=1):
            gr_variable_1 = gr.Code(interactive=False, wrap_lines=True, label="gr_variable_1", elem_id="gr_variable_1")
            gr_var_prompt = gr.Code(interactive=False, wrap_lines=True, label="gr_var_prompt", elem_id="gr_var_prompt")
            gr_infopool = gr.Radio(choices=["DVC_TTHC_LamDong", "DVC_TTHC_LangSon"], elem_id="gr_infopool", value="DVC_TTHC_LamDong", type="value", info="### Nguồn dữ liệu", interactive=True, container=False)
        with gr.Column(elem_id="gr_mid_column", min_width=800):
            gr_history = gr.Chatbot(elem_id="gr_history", type="messages", placeholder="![image](https://raw.githubusercontent.com/baobuiquang/DVC_SearchAssistant_V2/refs/heads/main/static/logo.png)\n## Xin chào!\nMình là chatbot hỗ trợ tìm kiếm thủ tục dịch vụ công.", avatar_images=(None, "https://raw.githubusercontent.com/baobuiquang/DVC_SearchAssistant_V3/refs/heads/main/static/logo.png"), group_consecutive_messages=False, container=False, examples=[
                {"text": "Hướng dẫn sử dụng", "files": []},
                {"text": "cach nop ho so", "files": []},
                {"text": "Hướng dẫn làm thủ tục", "files": []},
                {"text": "Tra cứu hồ sơ", "files": []},
                {"text": "sđt đường dây nóng hỗ trợ", "files": []},
                {"text": "", "files": []},
                {"text": "Điều kiện đăng ký kết hôn là gì?", "files": []},
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

if __name__ == "__main__":
    demo.launch()