import base64
import json
import os
import tempfile
import uuid
from io import BytesIO
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

router = APIRouter()

_yolo_model = None
_mistral_client = None
_gemini_client = None

CLOTHING_PROMPT = """Analyze this cropped clothing item and return ONLY a JSON object with these fields:
{
  "type": "clothing type (e.g. shirt, jeans, jacket)",
  "color": ["primary color", "secondary color if any"],
  "pattern": "solid / striped / floral / plaid / graphic / other",
  "style": "casual / formal / athletic / streetwear / other",
  "material": "material if identifiable, else null",
  "season": ["suitable seasons from: spring, summer, autumn, winter"],
  "notes": "one short sentence of notable features or null"
}
Return only valid JSON with no markdown, no explanation."""

SCENE_PROMPT = """Analyze this clothing/fashion image and return ONLY a JSON object with these fields:
{
  "scene_description": "brief description of the overall outfit or scene",
  "items_count": <integer number of clothing items visible>,
  "overall_style": "overall style aesthetic",
  "occasion": "suitable occasion (e.g. casual, formal, business, athletic)",
  "season": ["suitable seasons from: spring, summer, autumn, winter"],
  "styling_notes": "one sentence on how items are styled together or null"
}
Return only valid JSON with no markdown, no explanation."""

GEMINI_IMAGE_PROMPT = """Generate an ultra-realistic, high-resolution product photograph from the provided RGBA image of a {category}. The final image must adhere strictly to the following directives:

1.  Photorealism and Materiality:
* Render the garment with soft, diffuse, and even professional studio lighting that accurately showcases the material's texture and finish.
* The fabric must be rendered with high fidelity to represent {fabric}. For example, silk should have a soft sheen, cotton should appear matte, and denim should show a clear twill weave.
* The color must be precisely maintained as {color}, matching the source image without any color shifting or creative interpretation.
* Create subtle, soft shadows on the garment itself to give it three-dimensional form, but avoid casting any hard shadows onto the background.

2.  Garment Integrity and Semantic Preservation:
* CRITICAL: Preserve the exact semantic and structural design of the source garment. The cut, shape, silhouette, seams, stitching, collar, cuffs, buttons, zippers, pockets, and any graphical or embroidered details from the input cutout must be perfectly and accurately replicated.
* Do not output the RGBA cutout given as an input.
* Do not add, remove, or alter any design elements. The output must be the exact same garment, simply rendered in a perfect, realistic state.

3.  Pose and Condition:
* The garment must be presented in a perfectly neutral, symmetrical, and centered pose, as if on an invisible "ghost mannequin" or in a perfect flat-lay.
* It should be floating in the air, with a clear three-dimensional presence.
* The garment's condition must be pristine: perfectly steamed and completely free of any wrinkles, crinkles, creases, or fabrication flaws. The surface should be smooth and immaculate.

4.  Final Image Composition:
* The output must be a single, isolated image of the complete garment in 3D!
* The background must be fully transparent (alpha channel enabled).
* Do not include any human models, mannequins, props, hangers, tags, or any other objects.

Negative Constraints (What to AVOID):

    NO artistic stylization, filters, or dramatic lighting.

    NO wrinkles, folds, or creases.

    NO asymmetry, twisting, or dynamic poses.

    NO changes to the garment's design, color, or details.

    NO visible background, floor, or surfaces.

    NO people, limbs, or mannequins."""


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        import torch
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            from ultralyticsplus import YOLO
            _yolo_model = YOLO("kesimeg/yolov8n-clothing-detection")
        finally:
            torch.load = _orig_load
    return _yolo_model


def _get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        from mistralai import Mistral
        from app.core.config import settings
        if not settings.mistral_api_key:
            raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not set in .env")
        _mistral_client = Mistral(api_key=settings.mistral_api_key)
    return _mistral_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        from app.core.config import settings
        if not settings.google_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in .env")
        _gemini_client = genai.Client(api_key=settings.google_api_key)
    return _gemini_client


def _encode_image_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _mistral_metadata(crop_img: Image.Image) -> dict:
    """Send a crop to Mistral to extract per-item clothing metadata."""
    from app.core.config import settings
    b64 = _encode_image_b64(crop_img)
    client = _get_mistral_client()
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": CLOTHING_PROMPT},
            ],
        }],
    )
    return _parse_json_response(response.choices[0].message.content)


def _mistral_scene_analysis(orig_img: Image.Image) -> dict:
    """Send the full original image to Mistral for scene-level analysis."""
    from app.core.config import settings
    b64 = _encode_image_b64(orig_img)
    client = _get_mistral_client()
    response = client.chat.complete(
        model=settings.mistral_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": SCENE_PROMPT},
            ],
        }],
    )
    return _parse_json_response(response.choices[0].message.content)


def _gemini_generate_image(crop_img: Image.Image, metadata: dict) -> bytes:
    """Generate a product photograph using Gemini, driven by Mistral metadata."""
    from google import genai
    from google.genai import types
    from app.core.config import settings

    category = metadata.get("type", "clothing item")
    fabric = metadata.get("material") or "fabric"
    color = metadata.get("color", [])
    if isinstance(color, list):
        color = ", ".join(color) if color else "unknown"

    prompt = GEMINI_IMAGE_PROMPT.format(category=category, fabric=fabric, color=color)

    buf = BytesIO()
    crop_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=settings.google_model,
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
            types.Part(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise RuntimeError("Gemini returned no image in response")


def _upload_to_storage(bucket: str, path: str, data: bytes) -> str:
    """Upload bytes to Supabase Storage and return the public URL."""
    from app.core.supabase import get_supabase
    sb = get_supabase()
    sb.storage.from_(bucket).upload(path, data, {"content-type": "image/png"})
    return sb.storage.from_(bucket).get_public_url(path)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/catalog/collection")
def get_collection():
    """Return all jobs with their associated crops, newest first."""
    from app.core.supabase import get_supabase
    sb = get_supabase()
    jobs = sb.table("jobs").select("*").order("created_at", desc=True).execute()
    crops = sb.table("crops").select("*").order("created_at", desc=True).execute()

    crops_by_job: dict[str, list] = {}
    for c in crops.data:
        crops_by_job.setdefault(c["job_id"], []).append(c)

    result = []
    for job in jobs.data:
        result.append({**job, "crops": crops_by_job.get(job["id"], [])})

    return {"collection": result}


@router.post("/catalog/analyze")
async def analyze_catalog(files: List[UploadFile] = File(...)):
    """
    Pipeline:
      1. YOLO detects and segments clothing items.
      2. Mistral extracts per-crop metadata and full-image scene analysis.
      3. Gemini generates a product photograph for each crop.
      4. Images uploaded to Supabase Storage; metadata inserted into Supabase DB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    from ultralyticsplus import render_result
    from app.core.supabase import get_supabase

    yolo = _get_yolo_model()
    sb = get_supabase()
    output = []

    for upload in files:
        contents = await upload.read()
        ext = os.path.splitext(upload.filename or "img")[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            results = yolo.predict(source=tmp_path, save=False, verbose=False)
            result = results[0]

            job_id = uuid.uuid4().hex

            # Render YOLO detection overlay and upload to Storage
            render = render_result(model=yolo, image=tmp_path, result=result)
            rendered_url = _upload_to_storage(
                "results", f"{job_id}/rendered.png", _pil_to_bytes(render)
            )

            orig = Image.open(tmp_path).convert("RGB")

            # Scene analysis of the full original image
            try:
                scene = _mistral_scene_analysis(orig)
            except Exception as exc:
                scene = {"error": str(exc)}

            # Insert job row
            sb.table("jobs").insert({
                "id": job_id,
                "filename": upload.filename,
                "rendered_url": rendered_url,
                "scene_analysis": scene,
            }).execute()

            crops = []
            for i, box in enumerate(result.boxes):
                label = result.names[int(box.cls)]
                conf = round(float(box.conf), 2)
                x1, y1, x2, y2 = (int(c) for c in box.xyxy[0].tolist())

                crop_img = orig.crop((x1, y1, x2, y2))
                filename = f"{i}_{label}.png"

                # Upload crop to Storage
                crop_url = _upload_to_storage(
                    "results", f"{job_id}/{filename}", _pil_to_bytes(crop_img)
                )

                # Mistral: per-crop clothing metadata
                try:
                    metadata = _mistral_metadata(crop_img)
                except Exception as exc:
                    metadata = {"error": str(exc)}

                # Gemini: generate product photograph from crop + metadata
                generated_url = None
                generated_error = None
                try:
                    img_bytes = _gemini_generate_image(crop_img, metadata)
                    generated_url = _upload_to_storage(
                        "generated", f"{job_id}/{filename}", img_bytes
                    )
                except Exception as exc:
                    generated_error = str(exc)

                # Insert crop row
                sb.table("crops").insert({
                    "job_id": job_id,
                    "label": label,
                    "confidence": conf,
                    "crop_url": crop_url,
                    "generated_url": generated_url,
                    "generated_error": generated_error,
                    "metadata": metadata,
                }).execute()

                crops.append({
                    "label":           label,
                    "confidence":      conf,
                    "url":             crop_url,
                    "generated_url":   generated_url,
                    "generated_error": generated_error,
                    "metadata":        metadata,
                })

            crops.sort(key=lambda c: c["confidence"], reverse=True)

            output.append({
                "filename":       upload.filename,
                "rendered_url":   rendered_url,
                "scene_analysis": scene,
                "crops":          crops,
            })

        finally:
            os.unlink(tmp_path)

    return {"results": output}
