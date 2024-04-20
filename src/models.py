import whisperx
from omegaconf import DictConfig
from faster_whisper import WhisperModel
from helpers import find_numeral_symbol_tokens, wav2vec2_langs
from deepmultilingualpunctuation import PunctuationModel


class ModelsManager:

    def __init__(self, conf: DictConfig):

        self.mconf = conf.model_config
        self.mtype = self.mconf.mtypes[self.mconf.device]
        self.whisperx_model = whisperx.load_model(
            self.mconf.model_name,
            self.mconf.device,
            compute_type=self.mtype,
            asr_options={"suppress_numerals": self.mconf.suppress_numerals},
        )
        self.whisper_model = WhisperModel(self.mconf.model_name, device=self.mconf.device, compute_type=self.mtype)

        self.punct_model = PunctuationModel(model="kredor/punctuate-all")

    def punctuate(self, words_list):
        return self.punct_model.predict(words_list)
    def transcribe_batched(
            self,
            audio_file: str,
    ):
        audio = whisperx.load_audio(audio_file)
        result = self.whisperx_model.transcribe(audio, language=self.mconf.language, batch_size=self.mconf.batch_size)
        return result["segments"], result["language"]

    def transcribe(
            self,
            audio_file: str,

    ):

        language = self.mconf.language
        suppress_numerals = self.mconf.suppress_numerals

        if suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.whisper_model.hf_tokenizer)
        else:
            numeral_symbol_tokens = None

        if language is not None and language in wav2vec2_langs:
            word_timestamps = False
        else:
            word_timestamps = True

        segments, info = self.whisper_model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            word_timestamps=word_timestamps,
            suppress_tokens=numeral_symbol_tokens,
            vad_filter=True,
        )
        whisper_results = []
        for segment in segments:
            whisper_results.append(segment._asdict())

        return whisper_results, info.language

    def get_word_timestamps(self, whisper_results, vocal_target, language):

        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device=self.mconf.device
        )

        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, self.mconf.device
        )
        word_timestamps = self.filter_missing_timestamps(
            result_aligned["word_segments"],
            initial_timestamp=whisper_results[0].get("start"),
            final_timestamp=whisper_results[-1].get("end"),
        )
        return word_timestamps
        # clear gpu vram

    def filter_missing_timestamps(
            self,word_timestamps, initial_timestamp=0, final_timestamp=None
    ):
        # handle the first and last word
        if word_timestamps[0].get("start") is None:
            word_timestamps[0]["start"] = (
                initial_timestamp if initial_timestamp is not None else 0
            )
            word_timestamps[0]["end"] = self._get_next_start_timestamp(
                word_timestamps, 0, final_timestamp
            )

        result = [
            word_timestamps[0],
        ]

        for i, ws in enumerate(word_timestamps[1:], start=1):
            # if ws doesn't have a start and end
            # use the previous end as start and next start as end
            if ws.get("start") is None and ws.get("word") is not None:
                ws["start"] = word_timestamps[i - 1]["end"]
                ws["end"] = self._get_next_start_timestamp(word_timestamps, i, final_timestamp)

            if ws["word"] is not None:
                result.append(ws)
        return result

    def _get_next_start_timestamp(self,word_timestamps, current_word_index, final_timestamp):
        # if current word is the last word
        if current_word_index == len(word_timestamps) - 1:
            return word_timestamps[current_word_index]["start"]

        next_word_index = current_word_index + 1
        while current_word_index < len(word_timestamps) - 1:
            if word_timestamps[next_word_index].get("start") is None:
                # if next word doesn't have a start timestamp
                # merge it with the current word and delete it
                word_timestamps[current_word_index]["word"] += (
                        " " + word_timestamps[next_word_index]["word"]
                )

                word_timestamps[next_word_index]["word"] = None
                next_word_index += 1
                if next_word_index == len(word_timestamps):
                    return final_timestamp

            else:
                return word_timestamps[next_word_index]["start"]
