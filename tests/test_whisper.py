import whisper
import torch 
import sys 
from pathlib import Path 

model = whisper.load_model("small")
result = model.transcribe("./japanese.mp3", task='translate')
print(result["text"])
