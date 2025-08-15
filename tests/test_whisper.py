import whisper
import torch 
import sys 
from pathlib import Path 

model = whisper.load_model("small")
result = model.transcribe("./files/english.wav", verbose=True)
print(result["text"])
sys.exit(0)
result = model.transcribe("./japanese.mp3", task='translate')
print(result["text"])
result = model.transcribe("./yourself.mp3")
