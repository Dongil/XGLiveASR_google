import gradio as gr
from gradio_i18n import Translate, gettext as _

with gr.Blocks() as app:
    lang = gr.Radio(choices=["en", "ko"], value="ko", label="Language", visible=True)

    with Translate("configs/translation.yaml", lang=lang):
        with gr.Column():
            gr.Files(label=_("Upload File here"))  # ✅ 실제 레이아웃 컨텍스트에서 사용되도록 보장

app.launch()
