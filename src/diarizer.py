import contextlib
import re
import tempfile

import demucs.separate
from omegaconf import DictConfig
from helpers import wav2vec2_langs, get_words_speaker_mapping, punct_model_langs, \
    get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping
from models import ModelsManager
from nemo_process import NemoDiarizer
from icecream import ic
import os


class ParallelDiarizer:
    def __init__(self, audio_path: str, job_id: int, user_id: int, prompt_template_id: int,
                 config: DictConfig,
                 models: ModelsManager,
                 function_mode: str = None
                 ):
        ic("!!!!!")
        ic(audio_path)
        self.function_mode = function_mode
        self.audio_path: str = audio_path
        self.task_id: int = job_id
        self.user_id: int = user_id
        self.prompt_template_id: int = prompt_template_id
        self.config: DictConfig = config
        self.models = models

        # Initialize temporary directory variable
        self._work_dir = None
        with self._create_temp_dir() as temp_dir:
            self._work_dir = temp_dir

        self.vocal_target = None

        self.stemming: bool = config.model_config.stemming
        self.suppress_numerals: bool = config.model_config.suppress_numerals
        self.model_name: str = config.model_config.model_name
        self.batch_size: int = config.model_config.batch_size
        self.device: str = config.model_config.device
        self.language: str = config.model_config.language

    @contextlib.contextmanager
    def _create_temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def start_diarize(self):
        print("Starting diarization process")
        if self.stemming:
            # TODO: Suriya is this something we need? Demucs only separates musical instruments and voice. Could this be more robust?
            # Isolate vocals from the rest of the audio
            return_code = demucs.separate.main(
                ["-n", "htdemucs", "--two-stems=vocals", self.audio_path, "-o", self._work_dir])

            if return_code != 0:
                print(
                    "Source splitting with htdemucs failed, using original audio file."
                )
                self.vocal_target = self.audio_path
            else:
                self.vocal_target = os.path.join(
                    self._work_dir,
                    "htdemucs",
                    os.path.splitext(os.path.basename(self.audio_path))[0],
                    "vocals.wav",
                )
        else:
            self.vocal_target = self.audio_path

        print(f'Starting Nemo process with vocal_target: , {self.vocal_target}')
        nemo_thread = NemoDiarizer(self.audio_path, self._work_dir)
        try:
            nemo_thread.start_diarize()
        except Exception as e:
            print(f'error: {e}')

        # Transcribe the audio file
        if self.batch_size != 0:
            whisper_results, language = self.models.transcribe_batched(
                self.vocal_target,
            )
        else:

            whisper_results, language = self.models.transcribe(
                self.vocal_target,
            )

        if language in wav2vec2_langs:
            word_timestamps = self.models.get_word_timestamps(whisper_results, self.vocal_target, language)

        else:
            assert (
                    self.batch_size == 0  # TODO: add a better check for word timestamps existence
            ), (
                f"Unsupported language: {language}, use --batch_size to 0"
                " to generate word timestamps using whisper directly and fix this error."
            )
            word_timestamps = []
            for segment in whisper_results:
                for word in segment["words"]:
                    word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

        speaker_ts = []

        # interpret the rttm files from Nemo
        with open(f'{self._work_dir}/pred_rttms/mono_file.rttm', "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if language in punct_model_langs:

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = self.models.punctuate(words_list)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                        word
                        and labeled_tuple[1] in ending_puncts
                        and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            print(
                f"Punctuation restoration is not available for {language} language. Using the original punctuation."
            )
        ic("here")
        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Create a temporary directory
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     # Generate the temporary file paths
        #     txt_file = os.path.join(temp_dir, "transcript.txt")
        #     srt_file = os.path.join(temp_dir, "subtitles.srt")
        #
        #     # Write the transcript to the temporary txt file
        #     with open(txt_file, "w", encoding="utf-8-sig") as f:
        #         get_speaker_aware_transcript(ssm, f)
        #
        #     # Write the subtitles to the temporary srt file
        #     with open(srt_file, "w", encoding="utf-8-sig") as srt:
        #         write_srt(ssm, srt)
        #
        #     # Read the binary content of the temporary files
        #     with open(srt_file, "r") as srt_file:
        #         srt_content = srt_file.read()
        #
        #     cur = conn.cursor()
        #     logger.info(srt_content)
        #     logger.info('srt wrote to db')
        #     cur.execute("UPDATE loft_poc_audiorecord SET srt = %s, job_status = %s WHERE id = %s",
        #                 (srt_content, 1, self.task_id))
        #     conn.commit()
        #     cur.close()
        #
        #     asyncio.run(
        #         execute_call_post_diarization(self.task_id, self.user_id, self.prompt_template_id, self.function_mode))
        ic("Done!!")
        return
