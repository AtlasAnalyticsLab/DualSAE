import argparse
import base64
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


PAPER_STYLE_DENSE_CAPTION_PROMPT = """
You are an expert image analyst and descriptive writer specializing in creating ”dense captions.” Your task is to generate a single, continuous paragraph of highly detailed and comprehensive text that describes the provided image. Your description must be objective and based solely on visual evidence.
Follow this multi-step process for your analysis:
1. Holistic Overview: Begin by establishing the overall scene. Describe the setting (e.g., urban street, natural landscape, indoor room), the time of day (e.g., midday, golden hour, night), the overall atmosphere or mood (e.g., bustling, serene, melancholic), and the general color palette.
2. Primary Subjects and Actions: Identify and describe the primary subject(s) in detail. If they are people, describe their apparent age, gender, clothing, posture, expression, and any actions they are performing. If they are objects or animals, describe their type, condition, color, and position. Describe the interactions between primary subjects.
3. Secondary Elements and Background: Detail the secondary subjects, significant objects, and the immediate background. Describe architectural elements, furniture, vehicles, flora, and fauna that populate the scene but are not the central focus. Describe their spatial relationship to the primary subjects.
4. Fine-Grained Details and Textures: Scrutinize the image for fine-grained details. Mention specific textures (e.g., the rough bark of a tree, the smooth surface of a metal table, the fabric weave of a coat), small, easily missed objects, text or symbols visible on signs or clothing, reflections in windows or water, and the quality of light and shadow (e.g., sharp, defined shadows indicating harsh light, or soft, diffuse light).
5. Synthesis and Composition: Conclude by synthesizing all observations. Briefly describe the photographic composition, such as the framing, perspective, and depth of field (e.g., a shallow depth of field blurring the background, a wide-angle shot capturing a vast landscape).
Formatting and Style Constraints:
- Output Format: Your entire output must be one single, continuous paragraph.
- No Line Breaks: Do not use any line breaks, newlines, or paragraph breaks (\n).
- Style: Write in a descriptive, objective, and formal tone.
- Exclusions:
- Do not start with phrases like ”This is an image of,” ”The picture shows,” or any similar introductory statement.
- Do not include personal opinions, judgments, or interpretations that are not directly supported by visual evidence.
- Do not use bullet points, lists, or headers in your final output. Your entire response must be the caption itself.
Example of Desired Output:
A vibrant and crowded marketplace unfolds under the bright, hazy sun of midday, characterized by a dominant palette of warm ochres, deep reds, and earthen browns. The central focus is a male vendor in his late fifties, wearing a light blue djellaba and a straw hat, who is carefully arranging a pyramid of colorful spices on a rough wooden stall; his face is weathered and creased in concentration. In front of him, a tourist with a backpack slung over one shoulder, clad in a khaki shirt, points at a specific spice mound while a young child clings to her hand, looking with wide eyes at a nearby stall selling intricately woven leather bags. The background is a dense tapestry of activity, with other shoppers and vendors creating a soft-focus blur of movement, set against the backdrop of ancient, reddish-pink plaster walls and arched doorways. Fine details abound, from the coarse texture of the burlap sacks holding grains and the gleam of polished brass lanterns hanging from a wooden beam, to the subtle shadows cast by the woven canopy overhead, dappling the ground in a shifting pattern of light. The composition is tight and layered, creating a deep sense of immersion and chaotic energy, capturing the scene from a slightly low, eye-level perspective that places the viewer directly within the bustling alleyway.
Your Task:
Now, analyze the following image and generate the dense caption, strictly adhering to all instructions above.
""".strip()


def load_prompt(prompt_file: str | None) -> str:
    return PAPER_STYLE_DENSE_CAPTION_PROMPT


def iter_images(root: Path):
    images = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            images.append(p)
    return sorted(images)


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_done(output_path: Path):
    done = set()

    if not output_path.exists():
        return done

    # errors="replace": the existing OpenImages caption file contains some invalid UTF-8 bytes;
    # strict decoding raises mid-iteration (outside the per-line try) and would abort resumption.
    # Replacing bad bytes lets us still recover the rel_path of every well-formed record for the
    # done-set. (Malformed records that don't json-parse are treated as not-done → recaptioned.)
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(obj["rel_path"])
            except Exception:
                continue

    return done

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def caption_batch(
    model,
    tokenizer,
    image_paths: list[Path],
    rel_paths: list[str],
    prompt: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    dataset: str = "imagenet",
):
    transform = build_transform(input_size=448)
    pixel_values_list = []
    num_patches_list = []
    kept_image_paths = []
    kept_rel_paths = []

    # Load images one-by-one and SKIP any that fail to open/preprocess (corrupt/truncated files),
    # so a single bad image cannot crash the whole batch. A skipped image just goes uncaptioned; a
    # re-run re-attempts (and re-skips) it.
    for image_path, rel_path in zip(image_paths, rel_paths):
        try:
            image = Image.open(image_path).convert('RGB')
            images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
            pixel_values = torch.stack([transform(im) for im in images]).to(torch.bfloat16).cuda()
        except Exception as e:
            print(f"  [skip] failed to load {rel_path}: {e}")
            continue
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
        kept_image_paths.append(image_path)
        kept_rel_paths.append(rel_path)

    if not pixel_values_list:
        return []   # whole batch unreadable → nothing to caption
    # Downstream (responses zip, error-results, class_folder) uses the kept, successfully-loaded subset.
    image_paths, rel_paths = kept_image_paths, kept_rel_paths

    pixel_values_combined = torch.cat(pixel_values_list, dim=0)
    questions = [prompt] * len(image_paths)
    
    # NOTE: generation samples (do_sample=True) with the model's own generation defaults; the
    # --temperature value is accepted by the CLI but is NOT forwarded here. This is how the captions
    # used for the topic models were produced, so it is kept unchanged.
    generation_config = dict(
        max_new_tokens=max_tokens, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    last_error = None
    for _ in range(retries):
        # try:
        responses = model.batch_chat(
            tokenizer,
            pixel_values_combined,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        results = []
        for i, (rel_path, caption) in enumerate(zip(rel_paths, responses)):
            result = {
                "rel_path": rel_path,
                "caption": clean_caption(caption),
                "model": getattr(model, "__class__", str(model)).__name__,
            }
            # ImageNet has class subfolders; OpenImage v7 is flat
            if dataset == "imagenet":
                result["class_folder"] = image_paths[i].parent.name
            results.append(result)
        return results

        # except Exception as e:
        #     last_error = str(e)
        #     time.sleep(min(2**attempt, 30))

    # If all retries fail
    error_results = []
    for i, rel_path in enumerate(rel_paths):
        result = {
            "rel_path": rel_path,
            "caption": None,
            "model": getattr(model, "__class__", str(model)).__name__,
            "error": last_error,
        }
        if dataset == "imagenet":
            result["class_folder"] = image_paths[i].parent.name
        error_results.append(result)
    return error_results


# Dataset-specific defaults
# Dataset-specific defaults, relative to $DATA_ROOT (default ./data). Every default can be overridden
# with --image-dir / --output.
_DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
DATASET_DEFAULTS = {
    "imagenet": {
        "image_dir": os.path.join(_DATA_ROOT, "imagenet", "val"),
    },
    "open_images": {
        "image_dirs": {
            "train": os.path.join(_DATA_ROOT, "open_image_v7", "train"),
            "valid": os.path.join(_DATA_ROOT, "open_image_v7", "valid"),
            "test": os.path.join(_DATA_ROOT, "open_image_v7", "test"),
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["imagenet", "open_images"], default="imagenet",
                        help="Dataset to generate captions for.")
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default=None,
                        help="Split to caption (required for open_images, ignored for imagenet).")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing images. Overrides dataset/split defaults.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path. Overrides dataset/split defaults.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Unused: the dense-caption prompt below is always used (kept for CLI compatibility).")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-8B", help="Model name from Hugging Face.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start", type=int, default=None, help="Start index of images to caption.")
    parser.add_argument("--end", type=int, default=None, help="End index of images to caption.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_HOME", "./assets/huggingface"),
        help="Hugging Face cache directory holding the (pre-downloaded) captioning model.",
    )
    parser.add_argument(
        "--resume-glob",
        type=str,
        default=None,
        help="Optional glob of caption JSONL files whose rel_paths are ALSO treated as done, so a "
             "parallel shard skips the GLOBAL done-set (the original split file + sibling shards). "
             "Combined with disjoint --start/--end windows this avoids recaptioning and append races.",
    )
    args = parser.parse_args()

    # Resolve image_dir and output from dataset/split defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.dataset == "open_images":
        if args.split is None:
            parser.error("--split is required for open_images dataset (train/valid/test).")
        if args.image_dir is None:
            args.image_dir = defaults["image_dirs"][args.split]
        if args.output is None:
            args.output = os.path.join(_DATA_ROOT, "captions", args.dataset,
                                       f"{args.split}_{os.path.basename(args.model)}_{args.start}_{args.end}.jsonl")
    else:  # imagenet
        if args.image_dir is None:
            args.image_dir = defaults["image_dir"]
        if args.output is None:
            args.output = os.path.join(_DATA_ROOT, "captions", args.dataset,
                                       f"{os.path.basename(args.model)}_{args.start}_{args.end}.jsonl")

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    prompt = load_prompt(args.prompt_file)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"Image dir: {image_dir}")
    print(f"Output: {output_path}")

    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        local_files_only=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, cache_dir=str(cache_dir))

    print("Using cuda:", torch.cuda.is_available())

    images = iter_images(image_dir)
    if args.start is not None and args.end is not None:
        images = images[args.start:args.end]

    done = read_done(output_path)
    # Seed the done-set from sibling caption files (the original split file + any other shards) so
    # this shard skips images already captioned elsewhere. Self-match (the output file) is a no-op.
    if args.resume_glob:
        import glob as _glob
        for _p in sorted(_glob.glob(args.resume_glob)):
            done |= read_done(Path(_p))
    todo = [(p, str(p.relative_to(image_dir))) for p in images]
    todo = [(p, rel) for p, rel in todo if rel not in done]

    print(f"Found {len(images)} images.")
    print(f"Already done: {len(done)}")
    print(f"Remaining: {len(todo)}")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create batches
    batches = []
    for i in range(0, len(todo), args.batch_size):
        batch_todo = todo[i:i + args.batch_size]
        image_paths_batch = [item[0] for item in batch_todo]
        rel_paths_batch = [item[1] for item in batch_todo]
        batches.append((image_paths_batch, rel_paths_batch))

    # Process batches and write results
    with open(output_path, "a", encoding="utf-8", buffering=1) as f:
        # Loop through batches and process
        for image_paths, rel_paths in tqdm(batches, desc="Processing batches"):
            results = caption_batch(
                model,
                tokenizer,
                image_paths,
                rel_paths,
                prompt,
                args.max_tokens,
                args.temperature,
                args.retries,
                dataset=args.dataset,
            )
            # Write batch results to file
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                # print(result)

main()