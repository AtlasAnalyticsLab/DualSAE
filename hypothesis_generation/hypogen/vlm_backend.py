"""Offline vision-language backend (InternVL2.5-8B via vLLM) for the interpretation + annotation steps.

Mirrors SAE-TM/evaluation.py's load-once vLLM pattern: a single LLM is built once and shared across all
prompts in a job. Two operations (the image analogs of HypotheSAEs' interpret + annotate):
  interpret(items)  : per (positive_images, negative_images) -> one-line natural-language hypothesis
  annotate(items)   : per (image, concept) -> bool (does the image satisfy the concept?)

Images are passed as base64 data URIs inside OpenAI-style chat messages; vLLM applies InternVL's chat
template and handles the image tokens. Images are downsized and InternVL dynamic tiling is capped
(mm_processor_kwargs.max_dynamic_patch) so that many-image interpretation prompts fit in max_model_len.

This module imports vllm lazily (only when a backend is instantiated) so the non-VLM stages never pay it.
"""

from __future__ import annotations

import base64
import io
import os
from typing import List, Optional, Tuple

from PIL import Image

from hypogen import config

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

DEFAULT_TASK_INSTRUCTIONS = (
    "The images are from ImageNet. Good features describe a specific visual element, for example:\n"
    '- "a dog with long, floppy ears"\n'
    '- "the metal spokes of a bicycle wheel"\n'
    '- "dense green foliage filling the background"\n'
    '- "bold red-and-white stripes"'
)


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), encoding="utf-8") as f:
        return f.read()


def pil_to_data_uri(img: Image.Image, max_side: int = 448, quality: int = 90) -> str:
    """Downsize to <=max_side and encode as a base64 JPEG data URI (keeps per-image token cost bounded)."""
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_feature(response: str) -> Optional[str]:
    """Extract the single quoted feature from a proposer response (mirrors HypotheSAEs parsing)."""
    if not response:
        return None
    response = response.strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    response = response.split("\n", 1)[0].strip()
    for prefix in ('- "', '-"', "- ", "-"):
        if response.startswith(prefix):
            response = response[len(prefix):]
            break
    return response.strip().strip('"').strip() or None


def parse_yesno(response: str) -> bool:
    """True iff the annotator's first token is an affirmative; default No when uncertain/empty."""
    if not response:
        return False
    first = response.strip().split()[0].lower().strip(".,:;\"'") if response.strip().split() else ""
    return first in ("yes", "y", "true")


def _img_part(uri: str) -> dict:
    return {"type": "image_url", "image_url": {"url": uri}}


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class VLM:
    """Single shared InternVL2.5-8B engine (offline vLLM)."""

    def __init__(self, model: str = config.VLM_MODEL,
                 max_images: int = config.VLM_MAX_IMAGES_PER_PROMPT,
                 max_model_len: int = config.VLM_MAX_MODEL_LEN,
                 gpu_mem: float = 0.9, max_dynamic_patch: int = 1,
                 image_max_side: int = 448, seed: int = 0):
        # Run the vLLM V1 engine in-process: on some systems the multiprocessing EngineCore
        # subprocess can hang at bootstrap (before loading weights) waiting on its READY handshake.
        # In-process is correct for this synchronous batched LLM.chat() use. Must precede the vllm
        # import; setdefault so an explicit environment export still wins.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM  # lazy import (heavy)

        self.image_max_side = image_max_side
        common = dict(model=model, trust_remote_code=True, dtype="auto",
                      max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                      limit_mm_per_prompt={"image": max_images}, seed=seed)
        # Cap InternVL dynamic tiling so multi-image prompts stay within max_model_len. The kwarg name
        # can vary by vLLM version; fall back to no mm_processor_kwargs if it is rejected.
        try:
            self.llm = LLM(**common, mm_processor_kwargs={"max_dynamic_patch": max_dynamic_patch})
        except (TypeError, ValueError) as e:
            print(f"[VLM] mm_processor_kwargs rejected ({e}); retrying without tiling cap.")
            self.llm = LLM(**common)

    def _generate(self, conversations: List[list], max_tokens: int, temperature: float) -> List[str]:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self.llm.chat(conversations, sp, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

    def _interpret_conv(self, pos_imgs: List[Image.Image], neg_imgs: List[Image.Image],
                        task_instructions: str) -> list:
        content = [_text_part("POSITIVE images (the feature detector fires strongly on these):")]
        content += [_img_part(pil_to_data_uri(im, self.image_max_side)) for im in pos_imgs]
        content.append(_text_part("NEGATIVE images (the feature detector does not fire on these):"))
        content += [_img_part(pil_to_data_uri(im, self.image_max_side)) for im in neg_imgs]
        content.append(_text_part(_load_prompt("interpret-latent-vision.txt").format(
            task_specific_instructions=task_instructions)))
        return [{"role": "user", "content": content}]

    def _annotate_conv(self, img: Image.Image, concept: str) -> list:
        content = [_img_part(pil_to_data_uri(img, self.image_max_side)),
                   _text_part(_load_prompt("annotate-vision.txt").format(concept=concept))]
        return [{"role": "user", "content": content}]

    def interpret(self, items: List[Tuple[List[Image.Image], List[Image.Image]]],
                  task_instructions: str = DEFAULT_TASK_INSTRUCTIONS,
                  max_tokens: int = 64, temperature: float = 0.7) -> List[Optional[str]]:
        """items: [(positive_images, negative_images), ...] -> [hypothesis_string_or_None, ...]."""
        if not items:
            return []
        convs = [self._interpret_conv(pos, neg, task_instructions) for pos, neg in items]
        return [parse_feature(r) for r in self._generate(convs, max_tokens, temperature)]

    def annotate(self, items: List[Tuple[Image.Image, str]],
                 max_tokens: int = 8, temperature: float = 0.0) -> List[bool]:
        """items: [(image, concept), ...] -> [bool, ...] (concept present in image?)."""
        if not items:
            return []
        convs = [self._annotate_conv(img, concept) for img, concept in items]
        return [parse_yesno(r) for r in self._generate(convs, max_tokens, temperature)]
