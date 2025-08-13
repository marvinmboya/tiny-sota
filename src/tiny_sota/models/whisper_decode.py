import torch 
import torch.nn.functional as F
from torch import inf 
from torch.distributions import Categorical
import zlib 

from .whisper_load import get_tokenizer 
from .whisper_meta import (
    LogitFilter, 
    SequenceRanker, 
    TokenDecoder,
    CHUNK_LENGTH,
    DecodingResult
) 

def verify_options(options):
    if (options.beam_size is not None 
        and options.best_of is not None):
        raise ValueError("beam_size and best_of can't be given together")
    if options.temperature == 0:
        if options.best_of is not None:
            raise ValueError("best_of with greedy sampling (T=0) is not compatible")
    if options.patience is not None and options.beam_size is None:
        raise ValueError("patience requires beam_size to be given")
    if options.length_penalty is not None and not (
        0 <= options.length_penalty <= 1
    ):
        raise ValueError("length_penalty (alpha) should be a value between 0 and 1")
    return options

def get_initial_tokens(
        tokenizer, sot_sequence, 
        sample_len, n_ctx, options):
    tokens = list(sot_sequence)
    if prefix := options.prefix:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip())
            if isinstance(prefix, str)
            else prefix)
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens
    if prompt := options.prompt:
        prompt_tokens = (
            tokenizer.encode(" " + prompt.strip())
            if isinstance(prompt, str)
            else prompt)
        tokens = ([tokenizer.sot_prev]
            + prompt_tokens[-(n_ctx // 2 - 1) :]
            + tokens)
    return tuple(tokens)

def get_suppress_tokens(tokenizer, options):
    suppress_tokens = options.suppress_tokens
    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []
    else:
        assert isinstance(suppress_tokens, list), \
        "suppress_tokens must be a list"
    suppress_tokens.extend([tokenizer.transcribe,
            tokenizer.translate, tokenizer.sot,
            tokenizer.sot_prev, tokenizer.sot_lm])
    if tokenizer.no_speech is not None:
        suppress_tokens.append(tokenizer.no_speech)
    return tuple(sorted(set(suppress_tokens)))

def get_audio_features(model, mel, options):
    if options.fp16:
        mel = mel.half()
    if mel.shape[-2:] == (
        options.n_audio_ctx,
        options.n_audio_state):
        audio_features = mel
    else:
        audio_features = model.encoder(mel)
    if audio_features.dtype != (torch.float16 
        if options.fp16 else torch.float32):
        return TypeError(
            f"audio_features has an incorrect dtype: {audio_features.dtype}"
        )
    return audio_features

@torch.no_grad()
def model_detect_language(
        model, mel, tokenizer, config, 
        decode_options, init_tok_len):
    if tokenizer is None:
        tokenizer = get_tokenizer(
            language = decode_options.language,
            task = decode_options.task,
            is_multilingual = decode_options.is_multilingual)
    if (tokenizer.language is None
        or tokenizer.language_token 
        not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    if mel.shape[-2:] != (config.n_audio_ctx, config.n_audio_state):
        mel = model.encoder(mel)
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = inference(model, x, mel, init_tok_len)[:, 0]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -torch.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        { c: language_token_probs[i, j].item()
         for j, c in zip(
            tokenizer.all_language_tokens, 
            tokenizer.all_language_codes
        )} for i in range(n_audio)]
    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]
    return language_tokens, language_probs

def detect_language(
     *, model, audio_features, tokenizer, tokens, 
    config, options, sot_index, init_tok_len
    ):
    language = options.language
    task = options.task
    languages = [language] * audio_features.shape[0]
    lang_probs = None
    if language is None or task == "lang_id":
        lang_tokens, lang_probs = model_detect_language(
            model, audio_features, tokenizer, 
            config, options, init_tok_len
        )
        languages = [max(probs, key=probs.get) for probs in lang_probs]
        if language is None:
            tokens[:,sot_index+1] = lang_tokens
    return languages, lang_probs

def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer, sample_begin):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tokens.shape[1]==self.sample_begin:
            logits[:,self.tokenizer.encode(" ")+[self.tokenizer.eot]] = -inf

class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        logits[:, self.suppress_tokens] = -inf

class MaximumLikelihoodRanker(SequenceRanker):
    def __init__(self, length_penalty):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result
        
        lengths = [[len(t) for t in s] for s in tokens]
        return [torch.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]

class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature, eot):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens, logits, sum_logprobs):
        if self.temperature == 0:
            next_tokens = logits.torch.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens, sum_logprobs):
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()

class ApplyTimestampRules(LogitFilter):
    def __init__(self, tokenizer, sample_begin,
        max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        time_begin = self.tokenizer.timestamp_begin
        time_no = self.tokenizer.no_timestamps
        if time_no is not None:
            logits[:, time_no] = -inf
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin:]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = len(seq)>=1 and seq[-1]>=time_begin
            penultimate_was_timestamp = len(seq)<2 or seq[-2]>=time_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    logits[k, time_begin:] = -inf
                else:
                    logits[k, :self.tokenizer.eot] = -inf
            timestamps = sampled_tokens[sampled_tokens.ge(time_begin)]
            if timestamps.numel() > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, time_begin:timestamp_last] = -inf

        if tokens.shape[1] == self.sample_begin:
            logits[:, :time_begin] = -inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = time_begin + \
                    self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -inf
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, time_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, :time_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, :time_begin] = -inf

@torch.no_grad()
def inference(model, tokens, audio_features, init_tok_len):
    if tokens.shape[-1] > init_tok_len:
        tokens = tokens[:, -1:]
    return model.decoder(tokens, audio_features)

class DecodeTask:
    sequence_ranker = None
    decoder = None
    logit_filters = None
    def __init__(self, model, config, decode_options):
        options = decode_options
        self.model = model        
        language = getattr(options, "language", "en")
        task = getattr(options, "task", "transcribe")
        is_multilingual = getattr(options, "is_multilingual", True)
        tokenizer = get_tokenizer(
            language = language, task = task,
            is_multilingual = is_multilingual)
        self.tokenizer = tokenizer
        self.options = verify_options(options)
        
        self.n_group = options.beam_size or options.best_of or 1
        self.n_ctx = config.n_text_ctx
        self.sample_len = options.sample_len or config.n_text_ctx // 2
        self.sot_sequence = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps            
        self.initial_tokens = get_initial_tokens(
            self.tokenizer, self.sot_sequence, 
            self.sample_len, self.n_ctx, self.options
        )
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(tokenizer.sot)
        self.init_tok_len = len(self.initial_tokens)

        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)
        self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(
                  get_suppress_tokens(self.tokenizer,self.options)
              )
            )
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / config.n_audio_ctx
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _main_loop(self, audio_features, tokens):
        n_batch = tokens.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [torch.nan] * n_batch

        for i in range(self.sample_len):
            logits = inference(self.model, tokens, audio_features, self.init_tok_len)
            if (i==0 and self.tokenizer.no_speech is not None):
                probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
            logits = logits[:, -1]

            for logit_filter in self.logit_filters:
                logit_filter.apply(logits, tokens)
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
            if completed or tokens.shape[-1] > self.n_ctx:
                break
        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel):
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = get_audio_features(mel, self.options)
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        languages, language_probs = detect_language(
            model = self.model, 
            audio_features = audio_features, 
            tokenizer = tokenizer, 
            tokens = tokens, 
            config = self.config,
            options = self.options, 
            spt_index = self.sot_index
        )
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, 
                    language=language, 
                    language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]
        
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio
        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        
        tokens = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")
        
        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]
