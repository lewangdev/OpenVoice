import os
import io
import uuid
import torch
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
from pydub import AudioSegment
from fastapi import FastAPI, Response
from typing import Optional
from pydantic import BaseModel

en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# load speaker embeddings
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

# This online demo mainly supports English and Chinese
supported_languages = ['zh', 'en']


def predict(text, style, audio_file_pth):
    # initialize a empty info
    text_hint = ''

    # first detect the input language
    language_predicted = langid.classify(text)[0].strip()  
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
        return (
            text_hint,
            None,
            None,
        )
   
    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ['default']:
            text_hint += f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
            return (
                text_hint,
                None,
                None,
            )

    else:
        tts_model = en_base_speaker_tts
        if style == 'default':
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = 'English'
        if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']:
            text_hint += f"[ERROR] The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']\n"
            return (
                text_hint,
                None,
                None,
            )

    speaker_wav = audio_file_pth

    if len(text) < 2:
        text_hint += "[ERROR] Please give a longer prompt text \n"
        return (
            text_hint,
            None,
            None,
        )
    if len(text) > 200:
        text_hint += "[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        return (
            text_hint,
            None,
            None,
        )
    
    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        return (
            text_hint,
            None,
            None,
        )

    src_path = f'{output_dir}/tmp-{uuid.uuid4()}.wav'
    tts_model.tts(text, src_path, speaker=style, language=language)

    save_path = f'{output_dir}/output-{uuid.uuid4()}.wav'
    # Run the tone color converter
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)

    text_hint += 'Get response successfully \n'

    return (
        text_hint,
        save_path,
        speaker_wav,
    )


app = FastAPI()


class SpeechRequest(BaseModel):
    input: str
    voice: str = 'lisa'
    prompt: Optional[str] = ''
    language: Optional[str] = 'zh_us'
    model: Optional[str] = 'OpenVoice'
    style: Optional[str] = 'default'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0
    volume: Optional[int] = 10


@app.post("/v1/audio/speech")
def text_to_speech(speechRequest: SpeechRequest):
    text = speechRequest.input
    style = speechRequest.style
    voice = speechRequest.voice
    volume = speechRequest.volume
    audio_file_pth = f'./voices/{voice}.wav'

    (text_hint, save_path, speaker_wav) = predict(text, style, audio_file_pth)
    if save_path is None:
        return None
    audio = AudioSegment.from_file(save_path)
    audio = audio + volume
    wav_buffer = io.BytesIO()
    response_format = speechRequest.response_format
    audio.export(wav_buffer, format=response_format)

    return Response(content=wav_buffer.getvalue(),
                    media_type=f"audio/{response_format}")
