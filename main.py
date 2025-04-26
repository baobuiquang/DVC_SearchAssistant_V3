from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI
import gradio as gr
import uvicorn
from DVC_SearchAssistant import DVC_SearchAssistant

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

GLOBAL_AUTHENKEY = "VNPT"

class SearchAssistantPlayload(BaseModel):
    authen_key : str = "VNPT"
    datapool: str = "DVC_TTHC_LamDong"
    input: str = "Tôi muốn thành lập công ty thì cần phải làm gì?"

# ENDPOINT: /api
@app.post("/api", tags=["Main API Endpoint"])
async def endpoint_api(playload:SearchAssistantPlayload):
    if GLOBAL_AUTHENKEY != playload.authen_key:
        return {"message": "🧱 Your request is not authenticated!"}
    else:
        try:
            # --------------------------------------------------
            return DVC_SearchAssistant(inputtext=playload.input, infopool_id=playload.datapool)
            # --------------------------------------------------
        except Exception as e: return {"message": f"⚠️ Error: {e}"}

with gr.Blocks() as demo:
    gr.Markdown("# Hello")

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

app = gr.mount_gradio_app(app, demo, path="/demo")
if __name__ == "__main__":
    print("> Docs: http://127.0.0.1:5005/docs")
    print("> Demo: http://127.0.0.1:5005/demo")
    uvicorn.run(app, host = "0.0.0.0", port = 5005)