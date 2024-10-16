import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values

def traslator(audio_file):

    try:
        model= whisper.load_model("base")
        result= model.transcribe(audio_file, language="Spanish", fp16=False) 
        transcription= result["text"]

    except Exception as e:
        raise gr.Error(f"Se ha producido un error transcribiendo el texto:{str(e)} ")

    print(f"La transcripcion original es: {transcription}")
    
    try:
        en_trascriptions=Translator(from_lang="es", to_lang="en").translate(transcription)

    except Exception as e:
        raise gr.Error(f"Se ha producido un error traducciendo el texto:{str(e)} ")

    print(f"La transcripcion en ingles es: {en_trascriptions}")
    
    return en_trascriptions

web=gr.Interface(
    fn=traslator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=gr.Label(
        label="Inlges"
    ),
    title="Traductor de Voz",
    description="Traductor con IA"
)
web.launch()



# Minuto 20   https://www.youtube.com/watch?v=pNtcTmCiXzw