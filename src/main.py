from diarizer import ParallelDiarizer
import torch
from models import ModelsManager
from omegaconf import OmegaConf
import os
from icecream import ic

conf = OmegaConf.load(os.getenv("CONFIG_FILE", "/app/configs/app.yaml"))
conf.model_config.device = "cuda" if torch.cuda.is_available() else "cpu"

models = ModelsManager(conf)


def run_diarization(job_id, user_id, prompt_template_id, audio_file_path, function_mode):
    try:
        pd = ParallelDiarizer(audio_file_path, job_id, user_id, prompt_template_id, conf, models, function_mode)
        pd.start_diarize()
    except Exception as e:
        ic( e)


if __name__ == "__main__":

    run_diarization(0, 0, 0, "/input/audio1.wav", "")
