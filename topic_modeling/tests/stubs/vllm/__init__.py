"""Offline stand-in for vLLM used ONLY by the smoke tests (no model, no GPU).

Implements the tiny surface the pipeline uses -- LLM(...).generate(prompts, params) /
LLM(...).chat(conversations, params) and SamplingParams(...) -- and returns deterministic
canned answers chosen from the prompt text, in the formats the real parsers expect.
"""
import hashlib
import re


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    def __init__(self, text):
        self.text = text


class _Output:
    def __init__(self, text):
        self.outputs = [_Completion(text)]


def _h(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _answer(prompt):
    if "coherence of the following list" in prompt:            # SAE-TM coherence
        return '{"rationale": "stub", "score": %d}' % (40 + _h(prompt) % 50)
    m = re.search(r"The words are: (.*?)\. Your response", prompt, re.S)
    if m:                                                        # SAE-TM intruder
        words = [w.strip() for w in m.group(1).split(",") if w.strip()]
        return words[_h(prompt) % len(words)] if words else "none"
    if "Classify the topic" in prompt:                          # topic annotation
        return "concrete" if _h(prompt) % 3 else "abstract"
    if "Summarize the following vision topic" in prompt:        # topic label
        return "Stub Label %d" % (_h(prompt) % 7)
    if "Two topic models" in prompt:                            # image-topic judge
        return ["A", "B", "TIE"][_h(prompt) % 3]
    return "TIE"


class LLM:
    def __init__(self, model=None, **kwargs):
        self.model = model

    def generate(self, prompts, sampling_params=None, **kwargs):
        return [_Output(_answer(p)) for p in prompts]

    def chat(self, conversations, sampling_params=None, **kwargs):
        outs = []
        for conv in conversations:
            parts = conv[-1]["content"]
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
            uris = "".join(p["image_url"]["url"][-64:] for p in parts if p.get("type") == "image_url")
            if "PROPERTY:" in text:                             # VLM annotation (yes/no)
                outs.append(_Output("Yes" if _h(text + uris) % 2 else "No"))
            else:                                               # VLM interpretation (one quoted feature)
                outs.append(_Output('"stub feature %d"' % (_h(uris) % 5)))
        return outs
