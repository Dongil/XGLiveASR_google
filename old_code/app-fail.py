import os
import argparse
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger

logger = get_logger()

class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def launch(self):
        with open("configs/translation.yaml", "r", encoding="utf-8") as f:
            print("📄 현재 번역 YAML 내용:")
            print(f.read())

        with self.app:
            lang = gr.Radio(
                choices=list(self.i18n.keys()),
                label="Language",
                interactive=True,
                value="ko",
                visible=False
            )

            with Translate("configs/translation.yaml", lang=lang):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 🎯 프로젝트 시작", elem_id="md_project")

                with gr.Tabs():
                    with gr.TabItem(_("File")):
                        with gr.Column():
                            gr.Textbox(label=_("Language"))
                            gr.Textbox(label=_("Model"))
                            gr.Textbox(label=_("Upload File here"))
                            gr.Textbox(label=_("Translate to English?"))

        self.app.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper_type', type=str, default="faster-whisper")
    parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True)
    parser.add_argument('--server_name', type=str, default=None)
    parser.add_argument('--server_port', type=int, default=None)
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--username', type=str, default=None)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--theme', type=str, default=None)
    parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True)
    parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True)
    parser.add_argument('--allowed_paths', type=str, default=None)
    parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True)
    parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True)
    parser.add_argument('--ssl_keyfile', type=str, default=None)
    parser.add_argument('--ssl_keyfile_password', type=str, default=None)
    parser.add_argument('--ssl_certfile', type=str, default=None)
    parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR)
    parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR)
    parser.add_argument('--insanely_fast_whisper_model_dir', type=str, default=INSANELY_FAST_WHISPER_MODELS_DIR)
    parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR)
    parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR)
    parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    _args = parser.parse_args()

    app = App(args=_args)
    app.launch()
