#!/usr/bin/env python3
"""
YOLOv8 clothing detection pipeline (box-based cutouts + Nano Banana generation).

- Uses ultralyticsplus YOLO for clothing detection.
- Outputs box-based masks and cutouts (no resizing or post-processing).
- Reuses the Gemini 2.5 Flash Image ("Nano Banana") prompt and metadata logic.
"""

import argparse
import base64
import json
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from ultralyticsplus import YOLO
from ultralytics.nn.tasks import DetectionModel
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.modules.block as ul_block
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.container import Sequential, ModuleList
import torch.serialization
import inspect

# Allowlist YOLO classes for PyTorch 2.6+ safe loading
# Align module paths with the checkpoint global names
Conv.__module__ = "ultralytics.nn.modules"
C2f.__module__ = "ultralytics.nn.modules"
for _, obj in ul_block.__dict__.items():
    if inspect.isclass(obj):
        obj.__module__ = "ultralytics.nn.modules"

# Disable weights_only default to avoid endless allowlisting (trusted YOLO weights)
try:
    torch.serialization._default_to_weights_only = lambda *args, **kwargs: False
except Exception:
    pass
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv, C2f, ul_modules.C2f])
torch.serialization.add_safe_globals([Conv2d])
torch.serialization.add_safe_globals([BatchNorm2d])
torch.serialization.add_safe_globals([SiLU])
torch.serialization.add_safe_globals([ModuleList])

# Broad allowlist for ultralytics module classes used in YOLO checkpoints
_ultra_classes = [obj for _, obj in ul_modules.__dict__.items() if inspect.isclass(obj)]
_ultra_classes += [obj for _, obj in ul_block.__dict__.items() if inspect.isclass(obj)]
torch.serialization.add_safe_globals(_ultra_classes)
torch.serialization.add_safe_globals([C2f])


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def unique_dir_if_exists(directory: str) -> str:
    if not os.path.exists(directory):
        return directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{directory}_{ts}"


def extract_metadata_field(text: str, field_name: str) -> Optional[str]:
    import re
    patterns = [
        rf"\*\*{field_name}:\*\*\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"\*\*{field_name}:\*\*\s*\n\s*\*\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"{field_name}:\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"{field_name}:\s*\n\s*\*\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"\*\s*{field_name}:\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"{field_name}:\s*\n\s*-\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"{field_name}:\s*\n\s*\d+\.\s*(.+?)(?=\n\s*\*\*|\n\s*\*|\n\s*$|$)",
        rf"{field_name}:\s*(.+?)(?=\n|$)",
        rf"{field_name}:\s*(?:Given the.*?resembles a\s*)?(.+?)(?=\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
            content = re.sub(r"\*([^*]+)\*", r"\1", content)
            content = re.sub(r"^\s*[-*]\s*", "", content)
            content = re.sub(r"^\s*\d+\.\s*", "", content)
            content = re.sub(r"\n.*$", "", content)
            content = re.sub(r"^\s*Given the.*?resembles a\s*", "", content)
            content = re.sub(r"^\s*The\s+", "", content)
            content = re.sub(r"^\s*This\s+", "", content)
            return content.strip()
    return None


def generate_from_bytes_in_memory(
    image_bytes: bytes,
    api_key: str,
    category: str = "clothing item",
    fabric: str = "unknown fabric",
    color: str = "unknown color",
) -> Optional[bytes]:
    """Generate image using Gemini 2.5 Flash Image (Nano Banana) and return bytes."""
    try:
        msg_image = types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/png"))
        prompt_text = f"""Professional product photograph of the image that is visible to you in the input image.
Treat each image as a separate context.

EXACT REPLICATION REQUIRED:
- Preserve the exact garment design, cut, silhouette, seams, stitching, collar, cuffs, buttons, pockets, and all details from the input image
- Maintain precise color matching.
- Keep identical fabric texture characteristic.

PRESENTATION:
- Perfectly symmetrical, centered, front-facing neutral pose
- Ghost mannequin effect - floating 3D garment, no visible support
- Pristine condition: completely wrinkle-free, smooth, professionally steamed
- Soft, even studio lighting with subtle shadows for depth
- Transparent background (PNG alpha channel)

PHOTOREALISM:
- Ultra-realistic fabric rendering with accurate material properties
- Professional e-commerce quality
- High resolution, sharp details

STRICT EXCLUSIONS:
No models, no mannequins, no hangers, no props, no background elements, no artistic"""
        msg_text = types.Part.from_text(text=prompt_text)

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[msg_image, msg_text],
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        return None
    except Exception:
        return None


def gemini_analysis_for_generation(
    generated_image_bytes: bytes,
    original_image_bytes: bytes,
    api_key: str,
) -> Tuple[str, str, str, str]:
    """Extract category, fabric, color, and full analysis text for dynamic prompt generation."""
    client = genai.Client(api_key=api_key)
    msg_generated_image = types.Part(inline_data=types.Blob(data=generated_image_bytes, mime_type="image/png"))
    msg_original_image = types.Part(inline_data=types.Blob(data=original_image_bytes, mime_type="image/png"))
    analysis_prompt = types.Part.from_text(text=f"""You are analyzing a generated garment image (first image) and comparing it with the original user photo (second image). Analyze the garment shape, silhoutte, size, texture, any patterns to compare with the images in the datasets and parameters you have been trained on and then using your knowledge, fashion styling expertise provide a three to four word description in each of the following fields. Ensure, that the accuracy of this data is of utmost importance to understand user style preferences and enhance their emotions to become their better self. We want to produce magic through the power of fashion for the user to write their story and make a mark.

    category of garment: Look at the GENERATED GARMENT IMAGE (first image) - analyze the garment's shape and size and body part it should be worn at to describe the garment as to what it is, so that it is easier for the user to search those exact words to search it online, for instance a romper, tank top, tshirt, cardigan etc. and then next to the category inside braces classify it into on of the following types: topwear, bottomwear, outerwear, shoes, accessories, bags dress, innerwear - use 3-4 words like romper (topwear), cardigan (outerwear) etc.
    Style: Look at the GENERATED GARMENT IMAGE (first image) - visually analyze the garment and come up with typical style the user is going for with this particular garment, for instance bohemian, chic, or boho-chic, modern etc. Analyze and give style description either standalone or combined in 3-4 words (only if it is possible or makes sense like boho-chic makes sense)
    Fabric of garment: Look at the GENERATED GARMENT IMAGE (first image) - analyze texture of garment and come up with the fabric and material properties in 3-4 words e.g., "soft organic cotton knit" or "crisp structured linen blend"
    Color of garment: Look at the GENERATED GARMENT IMAGE (first image) - visually analyze the garment and come up with the color or color pattern in 3-4 words e.g., "muted sage green" or "vibrant royal blue"
    Vibe to wear it in: Compare the GENERATED GARMENT IMAGE (first image) with the ORIGINAL USER PHOTO (second image) - look at the aesthetic style and compare how the garment fits the user's style for e.g., "casual relaxed comfortable" or "elegant sophisticated formal" in 3-4 words - NOT multiple different vibes combined
    Season to wear it in: Analyze the GENERATED GARMENT IMAGE (first image) and ORIGINAL USER PHOTO (second image) - consider fabric, color, style, category and user's context to give seasons to wear the garment in 3-4 words, e.g., "warm spring summer" or "cozy fall winter"
    The fit of the cloth: Compare the GENERATED GARMENT IMAGE (first image) with the ORIGINAL USER PHOTO (second image) - analyze the size of the garment and how it fits the user to describe the fit in 3-4 words as a single description, e.g., "slim tailored fitted" or "relaxed oversized comfortable" - NOT multiple different fits combined
    Stylist's notes: Look at both the GENERATED GARMENT IMAGE (first image) and ORIGINAL USER PHOTO (second image) - provide a brief description in 6-8 words about how the garment drapes on user's body and can be styled as a standalone piece and/or paired with user's style

IMPORTANT: Give the metadata as the description under the 7 categories stated without \\n, \\s, asterisks or space formatting. Just the title: description, title: description format. Each category should be a single cohesive 3-4 word description (not multiple options combined). Stylist's notes should be 6-8 words only.""")

    analysis_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[msg_generated_image, msg_original_image, analysis_prompt],
    )

    category = "clothing item"
    fabric = "unknown fabric"
    color = "unknown color"
    full_response_text = ""
    for part in analysis_response.candidates[0].content.parts:
        if part.text is not None:
            full_response_text = part.text.strip()
            break

    if full_response_text:
        extracted_category = extract_metadata_field(full_response_text, "Category of garment")
        if extracted_category:
            category = extracted_category
        extracted_fabric = extract_metadata_field(full_response_text, "Fabric of garment")
        if extracted_fabric:
            fabric = extracted_fabric
        extracted_color = extract_metadata_field(full_response_text, "Color of garment")
        if extracted_color:
            color = extracted_color

    return category, fabric, color, full_response_text


def gemini_analysis(cutout: np.ndarray, api_key: str) -> str:
    """Analyze garment using Gemini 2.5 Flash (simple category for tile)."""
    client = genai.Client(api_key=api_key)
    cutout_bio = BytesIO()
    Image.fromarray(cutout, mode="RGBA").save(cutout_bio, format="PNG")
    msg_image = types.Part(inline_data=types.Blob(data=cutout_bio.getvalue(), mime_type="image/png"))
    prompt = types.Part.from_text(text="Describe the garment category in 3-6 words.")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[msg_image, prompt],
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            return part.text.strip()
    return "clothing item"


def save_box_mask_and_cutout(
    image_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    output_dir: str,
    label: str,
    idx: int,
) -> Tuple[str, str, bytes]:
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop from box")

    crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    mask = np.full((crop_rgba.shape[0], crop_rgba.shape[1]), 255, dtype=np.uint8)
    crop_rgba[:, :, 3] = mask

    mask_path = os.path.join(output_dir, f"{label}_{idx}_mask.png")
    cutout_path = os.path.join(output_dir, f"{label}_{idx}_cutout.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(cutout_path, crop_rgba)

    cutout_bio = BytesIO()
    Image.fromarray(crop_rgba, mode="RGBA").save(cutout_bio, format="PNG")
    return mask_path, cutout_path, cutout_bio.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 clothing pipeline (box-based cutouts + Nano Banana)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output_dir", default="output_yolov8", help="Output directory")
    parser.add_argument("--yolo_model", default="kesimeg/yolov8n-clothing-detection", help="YOLOv8 model id/path")
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--skip_metadata", action="store_true", help="Skip Gemini metadata analysis")
    args = parser.parse_args()

    output_dir = unique_dir_if_exists(args.output_dir)
    ensure_dir(output_dir)
    initial_dir = os.path.join(output_dir, "initial_detections")
    final_dir = os.path.join(output_dir, "final_detections")
    ensure_dir(initial_dir)
    ensure_dir(final_dir)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Could not load image: {args.image}")

    model = YOLO(args.yolo_model)
    results = model.predict(args.image)
    result = results[0]

    names = result.names if hasattr(result, "names") else {}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("No detections.")
        return

    mask_images_dict: Dict[str, bytes] = {}
    cutout_images_dict: Dict[str, bytes] = {}
    original_image_bytes = open(args.image, "rb").read()

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        label = names.get(cls_id, f"class_{cls_id}")
        box_xyxy = box.xyxy[0].cpu().numpy()
        _, _, cutout_bytes = save_box_mask_and_cutout(image_bgr, box_xyxy, final_dir, label, i)
        # Save mask/cutout bytes for generation
        mask_filename = f"{label}_{i}_mask.png"
        cutout_filename = f"{label}_{i}_cutout.png"
        mask_images_dict[mask_filename] = b""  # placeholder; masks are saved on disk
        cutout_images_dict[cutout_filename] = cutout_bytes

    if not args.api_key:
        print("⚠️ GEMINI_API_KEY not set; generation will fail.")

    generated_items = []
    for idx, (cutout_filename, cutout_bytes) in enumerate(cutout_images_dict.items(), start=1):
        generated_image_bytes = generate_from_bytes_in_memory(cutout_bytes, args.api_key)
        if not generated_image_bytes:
            continue
        # Metadata generation disabled (Gemini 2.5 Flash)
        category, fabric, color, analysis_text = "clothing item", "unknown fabric", "unknown color", ""
        tile_category = "clothing item"
        generated_items.append(
            {
                "generated_image_bytes": generated_image_bytes,
                "category": tile_category,
                "confidence": 0.9,
                "analysis": analysis_text,
                "metadata": {"category": category, "fabric": fabric, "color": color},
            }
        )

    generation_dir = os.path.join(output_dir, "generated_items")
    ensure_dir(generation_dir)
    for idx, item in enumerate(generated_items, start=1):
        image_path = os.path.join(generation_dir, f"generated_{idx}.png")
        with open(image_path, "wb") as img_f:
            img_f.write(item["generated_image_bytes"])
        metadata_path = os.path.join(generation_dir, f"generated_{idx}.json")
        with open(metadata_path, "w") as meta_f:
            json.dump(
                {
                    "category": item.get("category"),
                    "confidence": item.get("confidence"),
                    "analysis": item.get("analysis"),
                    "metadata": item.get("metadata"),
                },
                meta_f,
                indent=2,
            )

    print(f"✅ Saved cutouts to: {final_dir}")
    print(f"✅ Generated items saved to: {generation_dir}")


if __name__ == "__main__":
    main()
