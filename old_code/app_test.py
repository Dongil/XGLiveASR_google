import gradio as gr
from gradio_i18n import Translate, gettext as _

with gr.Blocks() as demo:
    lang = gr.Radio(choices=["en", "ko"], value="ko", label="Language", visible=True)

    with Translate("configs/translation.yaml", lang=lang):
        print("🔤 _(Language):", _("Language"))  # 출력: '언어'
        gr.Textbox(label=_("Language"))          # UI: '언어'
        gr.Textbox(label=_("Upload File here"))  # UI: '파일업로드'

demo.launch()
