import torch 
import torch.nn.functional as F
from numpy import inf 

from .configs import Audio_Mel_Params, DecodeOptions, DecodeResult

from .whisper_utils import (
    verify_options,
    get_initial_tokens,
    get_suppress_tokens,
    get_audio_features,
    compression_ratio,
    MaximumLikelihoodRanker,
    SuppressBlank,
    SuppressTokens,
    ApplyTimestampRules,
    GreedyDecoder,
    CachedInference
)

class DecodeTask:
    sequence_ranker = None
    decoder = None
    logit_filters = None
    def __init__(self, model, tokenizer, config, decode_options: DecodeOptions):
        options = decode_options
        self.model = model      
        self.config = config  
        task = getattr(options, "task", "transcribe")
        self.tokenizer = tokenizer
        self.options = verify_options(options)
        self.options.language = getattr(options, "language", "en") or "en"
        self.options.is_multilingual = getattr(options, "is_multilingual", True)
        
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
        self.inference = CachedInference(model, len(self.initial_tokens))

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
            precision = Audio_Mel_Params.CHUNK_LENGTH / config.n_audio_ctx
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
        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)
                if (i==0 and self.tokenizer.no_speech is not None):
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                logits = logits[:, -1]
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()
        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel):
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = get_audio_features(self.model, mel, self.config, self.options.dtype)
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        languages = [self.options.language] * audio_features.shape[0]
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

        fields = (texts, languages, tokens, 
                  audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")
        
        return [
            DecodeResult(
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
