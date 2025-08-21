# tiny-sota
> An (active development) neat inference library that implements and optimizes tiny latest release State-Of-The-Art models while decoupling them from the complex transformers library and aims to tune the inference pipelines across low-power peripheral devices and newer phones with SoC chips which form a major device of the ecosystem of physical AI (which will shift the AI age)

|      Model     | Domain | 🤗 Hub Size | Learning Impact |  Developed  |
|:----------|:--------------------:|:----------------:|:-------------------------:|:-------------------------------:|
| Qwen3-0.6B | 📖 to 📖 (Thinking Mode 🤔) |  1.5 GB  | Grouped-Query Attention   |     ✅      |
| Llama3.2-1B   |   📖 to 📖  |    2.47 GB      | From Muti-Head to Grouped-Query Attention |     ✅      |
| Whisper Small  |   💬 to 📖   |     461 MB      |  Pytorch Forwards Hooks Deep Dive |     ✅      |
| Kokoro |   📖 to 💬   |    312 MB  | Albert (& Bert)Transformers   |     ✅      |

### Next focuses
- YOLO, first single-stage pipeline object detection by Redmon et al., inspired my final year undergraduate project
- FastVLM, nice variant of Vision-Language Models built by the Apple Team focused on on-device inference

### Usage
(For now) clone the repository, run `pip install -e .` on the root directory\
safe to run, downloads state dict from HuggingFace Hub and the Azure official OpenAI repo (for whisper model), transfers those weights to the graphs implemented purely in `PyTorch`
### Example engine run
```py
from tiny_sota.inference import STTEngine
from tiny_sota.models import Whisper, ModelConfigs

from tiny_sota.models import loadWhisperSmallWeightsAndTok
from tiny_sota.models.configs import SpeechOptions
from tiny_sota.transfers import transferWhisperWeights
from tiny_sota.tiny_utils import get_device

device = get_device()
config = ModelConfigs.Whisper
model = Whisper(config)

weights, tok = loadWhisperSmallWeightsAndTok()
transferWhisperWeights(model, config, weights)

engine = STTEngine(model, tok, device)
speech_ops = SpeechOptions()

engine("./files/Spanish-greetings.mp3", speech_ops)
engine("./files/japanese.mp3", speech_ops)
engine.switch_task()
engine("./files/japanese.mp3", speech_ops)
engine.switch_task()
engine("./files/english.wav", speech_ops, verbose=True)
```
Output:
```console 
 1️⃣  Hola. Buenos días. Buenas tardes. Buenas noches. ¿Cómo está? ¿Cómo le va? ¿Cómo le ha ido? ¿Cómo ha estado? ¿Bien y tú? Todo bien, gracias. Todo tranquilo. ¿Qué tal? ¿Cómo estás? ¿Cómo te va? ¿Cómo te ha ido? ¿Cómo has estado? Por ahí. Más o menos. Ahí vamos. Ahí. Pasándola. Hasta pronto. Hasta luego. Hasta la próxima. Nos vemos. Chao. ¡Adiós!
 2️⃣  近くの良いレストランを推薦していただけますか?
task -> translate...
 3️⃣ Can you recommend a restaurant nearby?
task -> transcribe...
 4️⃣ [00:00.000 --> 00:03.500]  The boy was there when the sun rose.
[00:03.500 --> 00:06.500]  A rod is used to catch pink salmon.
[00:06.500 --> 00:10.000]  The source of the huge river is the clear spring.
[00:10.000 --> 00:13.000]  Kick the ball straight ãnd follow through.
[00:13.000 --> 00:16.000]  Help the woman get back to her feet.
[00:16.000 --> 00:19.000]  A pot of tea helps to pass the evening.
[00:19.000 --> 00:22.000]  Smokey fires lack flame and heat.
[00:22.000 --> 00:25.000]  The soft cushion broke the man's fall.
[00:25.000 --> 00:28.000]  The salt breeze came across the sea.
[00:28.000 --> 00:32.000]  The girl at the booth sold 50 bonds.
 The boy was there when the sun rose. A rod is used to catch pink salmon. The source of the huge river is the clear spring. Kick the ball straight and follow through. Help the woman get back to her feet. A pot of tea helps to pass the evening. Smokey fires lack flame and heat. The soft cushion broke the man's fall. The salt breeze came across the sea. The girl at the booth sold 50 bonds.
```

## Key Optimization Strategies 
#### 🔥 Regional Compilation 
- shoutout to [@RisingSayak](https://x.com/RisingSayak/status/1942967151943618806) for the Flux.1-Dev inference speed-up post that reduced compile time by 7.5x.
- 📝 Idea: Instead of compiling the entire model at once, compile the repeated regions.
- ✅ Benefit: Faster compilation & quicker warm-up. 
#### 🔥 KV Cache
- 🔗 Excellent reads by [Raschka](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) and the [HuggingFace team](https://huggingface.co/blog/kv-cache) that race tightly for the win with their awesome articles!
#### 🔥 Torch compile
- nothing beats a [tutorial series](https://youtu.be/VNYBgqGQ98E?si=UeArLek3YDAVAbv2) for Torch Compile that teaches graph transformations into optimized static execution graphs from the PyTorch team itself
#### 🔥 Torch compile full graph True
- sweet source on top of torch compile
#### 🔥 Streaming LLMs
- uses a concept called attention sinks that is so good, OpenAI adopted it for their August release [GPT-OSS](https://openai.com/index/introducing-gpt-oss/) models.
#### 🔥 Operator fusion
- 🔗 Combine multiple operations (like matmul + bias add + activation) into one kernel launch.
- ✅ Saves memory bandwidth + reduces GPU launch overhead
#### 🔥 Speculative Decoding
- requires larger memory for running two models (one small, one large), not feasible for this project
#### 🔥  Quantization
- Dynamic quantization first release by the (zero-day because they keep the AI community engaged with neat updates) [Unsloth Team](https://unsloth.ai/blog/dynamic-4bit), rapid team with neat and cool posts, like their website so much!
- I and Q quantization strategies by the [llama.cpp](https://github.com/ggml-org/llama.cpp) team, shoutout to [Georgi Gerganov](https://x.com/ggerganov), the lead maintainer of the tool, and to Julia Turc for nicely documenting [these](https://github.com/iuliaturc/gguf-docs)   
#### 🔥 Low-level C development
- [Justine Tunney](https://x.com/justineTunney) wows the community with the [llamafile](https://github.com/Mozilla-Ocho/llamafile), a single-file large language models executable tool!
#### 🔥 GGUF models inference
🐎 GGUF format designed for efficient quantized inference, both used by llama.cpp and llamafile for their high-speedup cpu and gpu inferencing
- ✅ Portable across CPU/GPU, widely used with llama.cpp.
#### 🔥 CPU mmap and blocked matrix multiplications
- 🗂️ mmap allows loading large weights on-demand instead of fully into memory.
- 🔲 Blocked matrix multiplications (matrix tiling) improves cache locality, with a nice example of a [Triton implementation](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html) of the same.
- ✅ Boosts CPU inference speed drastically.

### Acknowledgements
- 🏆 [Sebastian Raschka](https://x.com/rasbt?lang=en) for from-scratch implementations of these powerful pioneering open-source models, particular Qwen and Llama models.
- 🏆 [OpenAI](https://github.com/openai/whisper/tree/main) for open-sourcing whisper group of models that opened up the speech to text modality for active research, a true robust model at work.
- 🛠️ [@hexgrad](https://huggingface.co/hexgrad
) for open-sourcing Kokoro, a magic of its own in inferencing, with cool Audio Post-processing techniques for audio generations and cool thought out workflows to make this work, including a tool release, G2P engine.
- ❤️ Special thanks to HuggingFace🤗 since most obstacles I got from implementations were resolved by going through their module calls and workflows, as much as it was very abstracted, massive respect!