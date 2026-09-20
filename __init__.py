"""RelateAnything v0.6: local, verified ONNX inference; no checkpoint loading.

Contract verified against Maelic/RelateAnything 6b9c07c12ff47d4be17fa558b183023dcf829e61
and maelic/relsgg-vits16plus caf70d3af46478614209d86881a16e601d6bbf61.
See README.md and evidence/. Node identifiers are distinct from the old torch nodes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

VERSION = "0.6.0"
DEFAULT_MODEL = "F:/ComfyUI/models/relateanything/relsgg-vits16plus/relateanything.onnx"
VERIFIED = {
    "relateanything.onnx": "b8b6a047c5e0771a897a5015c2ffb09d8fe5e3ffa0651e1e61af0e8436a3617a",
    "relateanything.json": "a5aa12ccd217337d6fdfff068943348e010ffec44035ee1b2a99d27085e34b14",
    "predicate_bank.npz": "708f812d6579eab1be85b3378b79e376453c2d4f65b4fad9bcebb9f5bf05da06",
}
CONTRACT = "sigmoid(a * (pred_logit + w * pair_logit) + b)"
INPUT_SPEC = {
    "image": ("tensor(float)", (None, 3, 448, 448)),
    "boxes": ("tensor(float)", (None, None, 4)),
    "box_counts": ("tensor(int64)", (None,)),
    "W": ("tensor(float)", (None, 512)),
    "alpha": ("tensor(float)", (None,)),
}
OUTPUT_SPEC = {
    "pred_logits": ("tensor(float)", (None, None, None)),
    "pair_logits": ("tensor(float)", (None, None)),
    "sub_idx": ("tensor(int64)", (None, None)),
    "obj_idx": ("tensor(int64)", (None, None)),
    "valid_mask": ("tensor(bool)", (None, None)),
}
DEFAULT_VOCAB = "\n".join([
    "in front of", "behind", "to the left of", "to the right of", "above", "below",
    "inside", "contains", "attached to", "standing on", "parked on", "next to",
    "reflected in", "casting shadow on", "supporting", "resting on", "part of",
])


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array(value):
    # Accept existing Comfy tensors without importing torch or loading any weights.
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _one(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name}: expected one source image/value; got {len(value)}.")
        return value[0]
    return value


def _image(value):
    x = _array(_one(value, "IMAGE"))
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != 3:
        raise ValueError(f"IMAGE must be one RGB image [1,H,W,3], got {x.shape}.")
    if min(x.shape[1:3]) < 1 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("IMAGE must be finite RGB values in [0,1].")
    return x[0]


def _masks(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _masks(item)
        return
    x = _array(value)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"MASK must be [H,W], [N,H,W], [N,1,H,W] or a list; got {x.shape}.")
    for m in x:
        if not np.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise ValueError("MASK values must be finite and in [0,1].")
        yield m


def _validate_regions(regions, h, w):
    if not isinstance(regions, dict) or not {"boxes", "height", "width"} <= regions.keys():
        raise ValueError("RA_REGIONS requires boxes [N,4] in pixel xyxy, height and width.")
    if (regions["height"], regions["width"]) != (h, w):
        raise ValueError(f"Source IMAGE is {w}x{h}; RA_REGIONS belongs to {regions['width']}x{regions['height']}.")
    b = np.asarray(regions["boxes"], dtype=np.float32)
    if b.size == 0:
        b = b.reshape(0, 4)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError("Boxes must be finite pixel xyxy [N,4].")
    if len(b) and (np.any(b[:, 2:] <= b[:, :2]) or np.any(b < 0)
                   or np.any(b[:, [0, 2]] > w) or np.any(b[:, [1, 3]] > h)):
        raise ValueError("Boxes must have positive area and lie inside the source IMAGE.")
    if len(b) > 32:
        raise ValueError(f"Verified export supports 32 regions; got {len(b)}. Filter explicitly upstream.")
    ids = regions.get("source_indices", list(range(len(b))))
    if len(ids) != len(b):
        raise ValueError("source_indices length must match boxes.")
    return np.ascontiguousarray(b), ids


def collect_regions(image, masks=None, boxes_json=None, threshold=0.5, min_area_px=20, max_regions=32):
    rgb = _image(image)
    h, w = rgb.shape[:2]
    if (masks is None) == (boxes_json is None):
        raise ValueError("Connect exactly one: SAM3 masks OR boxes_json (pixel xyxy JSON).")
    if not 0 < threshold <= 1 or not 1 <= max_regions <= 32 or min_area_px < 1:
        raise ValueError("Invalid region settings: threshold (0,1], min_area>=1, max_regions 1..32.")
    boxes, ids, received, rejected = [], [], 0, 0
    if masks is not None:
        for idx, m in enumerate(_masks(masks)):
            received += 1
            if m.shape != (h, w):
                raise ValueError(f"MASK {idx} is {m.shape}; source IMAGE is {(h,w)}. No automatic resize.")
            ys, xs = np.where(m >= threshold)
            if len(xs) < min_area_px:
                rejected += 1
                continue
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)])
            ids.append(idx)
    else:
        raw = json.loads(str(_one(boxes_json, "boxes_json")))
        b, _ = _validate_regions_uncapped(raw, h, w)
        received = len(b)
        for idx, box in enumerate(b):
            if (box[2]-box[0]) * (box[3]-box[1]) < min_area_px:
                rejected += 1
                continue
            boxes.append(box.tolist())
            ids.append(idx)
    dropped = max(0, len(boxes) - max_regions)
    regions = {"boxes": np.asarray(boxes[:max_regions], dtype=np.float32).reshape(-1, 4),
               "height": h, "width": w, "source_indices": ids[:max_regions],
               "origin": "sam3_masks_to_boxes" if masks is not None else "sam3_boxes_xyxy"}
    _validate_regions(regions, h, w)
    debug = {"received": received, "accepted": len(regions["boxes"]), "filtered_small": rejected,
             "dropped_by_limit": dropped, "source_indices": regions["source_indices"],
             "boxes_xyxy": regions["boxes"].tolist(), "width": w, "height": h,
             "note": "One SAM3 instance mask = one box; input order preserved. ONNX has no mask input."}
    return regions, json.dumps(debug, ensure_ascii=False, indent=2)


def _validate_regions_uncapped(raw, h, w):
    # Validate all SAM3 boxes before explicit filtering/capping, never silently reinterpret coordinates.
    b = np.asarray(raw, dtype=np.float32)
    if b.size == 0:
        b = b.reshape(0, 4)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("SAM3 boxes_json must be a JSON array of pixel [x1,y1,x2,y2] boxes.")
    for start in range(0, len(b), 32):
        _validate_regions({"boxes": b[start:start+32], "height": h, "width": w}, h, w)
    return b, list(range(len(b)))


class RARegionsV06:
    INPUT_IS_LIST = True
    RETURN_TYPES = ("RA_REGIONS", "STRING")
    RETURN_NAMES = ("regions", "regions_debug")
    FUNCTION = "extract"
    CATEGORY = "RelateAnything/ONNX v0.6"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                             "min_area_px": ("INT", {"default": 20, "min": 1, "max": 100000000}),
                             "max_regions": ("INT", {"default": 32, "min": 1, "max": 32})},
                "optional": {"masks": ("MASK",), "boxes_json": ("STRING", {"forceInput": True})}}

    def extract(self, image, threshold, min_area_px, max_regions, masks=None, boxes_json=None):
        return collect_regions(image, masks, boxes_json, float(_one(threshold, "threshold")),
                               int(_one(min_area_px, "min_area_px")), int(_one(max_regions, "max_regions")))


def _signature(session):
    return {"inputs": [{"name": v.name, "type": v.type, "shape": v.shape} for v in session.get_inputs()],
            "outputs": [{"name": v.name, "type": v.type, "shape": v.shape} for v in session.get_outputs()]}


def _check_signature(signature):
    for direction, spec in (("inputs", INPUT_SPEC), ("outputs", OUTPUT_SPEC)):
        actual = {x["name"]: x for x in signature[direction]}
        if set(actual) != set(spec):
            raise ValueError(f"Unsupported ONNX {direction}: {list(actual)}; expected {list(spec)}.")
        for name, (dtype, shape) in spec.items():
            got = actual[name]
            if got["type"] != dtype or len(got["shape"]) != len(shape):
                raise ValueError(f"Unsupported type/rank for {name}: {got}.")
            for a, b in zip(got["shape"], shape):
                if b is not None and a != b:
                    raise ValueError(f"Unsupported shape for {name}: {got['shape']}.")


class RAOnnxLoadV06:
    RETURN_TYPES = ("RA_ONNX_MODEL", "STRING")
    RETURN_NAMES = ("ra_onnx_model", "diagnostics_text")
    FUNCTION = "load"
    CATEGORY = "RelateAnything/ONNX v0.6"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"onnx_path": ("STRING", {"default": DEFAULT_MODEL}),
                             "provider": (["CPUExecutionProvider", "CUDAExecutionProvider"],),
                             "threads": ("INT", {"default": 4, "min": 1, "max": 32})}}

    @classmethod
    def IS_CHANGED(cls, onnx_path, **kwargs):
        p = Path(os.path.expandvars(os.path.expanduser(onnx_path)))
        return tuple((str(x), x.stat().st_mtime_ns, x.stat().st_size) if x.is_file() else (str(x), None)
                     for x in (p, p.with_suffix(".json"), p.parent / "predicate_bank.npz"))

    def load(self, onnx_path, provider="CPUExecutionProvider", threads=4):
        import onnxruntime as ort
        p = Path(os.path.expandvars(os.path.expanduser(str(onnx_path).strip()))).resolve()
        if not p.is_file() or p.suffix.lower() != ".onnx":
            raise FileNotFoundError(f"Choose an existing local .onnx file: {p}. No automatic download.")
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} unavailable. Available: {ort.get_available_providers()}. Select CPU.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        session = ort.InferenceSession(str(p), sess_options=options, providers=[provider])
        if provider not in session.get_providers():
            raise RuntimeError(f"Requested {provider} failed to initialize: {session.get_providers()}.")
        signature = _signature(session)
        handle = {"session": session, "path": str(p), "signature": signature, "ready": False}
        report = {"version": VERSION, "onnx_path": str(p), "onnxruntime": ort.__version__,
                  "requested_provider": provider, "session_providers": session.get_providers(), **signature}
        try:
            _check_signature(signature)
            # Bind preprocessing and decoding to artifacts actually checked, not to guessed names alone.
            paths = {"relateanything.onnx": p, "relateanything.json": p.with_suffix(".json"),
                     "predicate_bank.npz": p.parent / "predicate_bank.npz"}
            report["sha256"] = {}
            for name, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Required sidecar missing: {path}")
                digest = _sha(path)
                report["sha256"][name] = digest
                if digest != VERIFIED[name]:
                    raise ValueError(f"Unverified artifact {name}, sha256={digest}. Diagnostic-only; use the pinned release.")
            meta = json.loads(paths["relateanything.json"].read_text(encoding="utf-8"))
            if (meta["score_contract"] != CONTRACT or meta["output_kind"] != "logits"
                    or meta["vocab_mode"] != "input" or meta["max_boxes"] != 32 or meta["img_size"] != 448):
                raise ValueError("Unsupported sidecar contract.")
            # Official names/default arrays are object dtype. Only the SHA256-verified official bank
            # is permitted to reach allow_pickle=True; unknown NPZ files never reach this line.
            with np.load(paths["predicate_bank.npz"], allow_pickle=True) as z:
                bank = {k: z[k].copy() for k in ("names", "default", "W", "alpha")}
            names = [str(n) for n in bank["names"]]
            if (bank["W"].shape != (len(names), 512) or bank["alpha"].shape != (len(names),)
                    or len(set(names)) != len(names) or not np.isfinite(bank["W"]).all()
                    or not np.isfinite(bank["alpha"]).all()):
                raise ValueError("Invalid predicate bank dimensions/values.")
            handle.update(ready=True, meta=meta, bank=bank, names=names)
            report.update(inference_ready=True, max_regions=32, available_predicates=names,
                          score_contract=CONTRACT, calibration=meta["calibration"],
                          note="SAM3 masks become boxes. No mask input in this ONNX export.")
        except (OSError, ValueError, KeyError) as exc:
            report.update(inference_ready=False, blocked_reason=str(exc))
            handle["blocked_reason"] = str(exc)
        handle["diagnostics"] = json.dumps(report, ensure_ascii=False, indent=2)
        return handle, handle["diagnostics"]


class RAOnnxInspectV06:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostics_text",)
    FUNCTION = "inspect"
    CATEGORY = "RelateAnything/ONNX v0.6"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ra_onnx_model": ("RA_ONNX_MODEL",)}}

    def inspect(self, ra_onnx_model):
        text = ra_onnx_model["diagnostics"]
        print("[RA ONNX diagnostic]\n" + text)
        return {"ui": {"text": [text]}, "result": (text,)}


def make_feed(rgb, boxes, names, model):
    import cv2  # Existing ComfyUI dependency, not installed/upgraded by this package.
    h, w = rgb.shape[:2]
    # Same uint8 RGB and INTER_LINEAR square resize as official runtime; no ImageNet normalization.
    u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    resized = cv2.resize(u8, (448, 448), interpolation=cv2.INTER_LINEAR)
    x = resized.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255.0)
    b = boxes.copy()
    b[:, [0, 2]] /= w
    b[:, [1, 3]] /= h
    normalized = np.stack(((b[:, 0]+b[:, 2])/2, (b[:, 1]+b[:, 3])/2,
                           b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]), axis=1)
    padded = np.zeros((1, 32, 4), dtype=np.float32)
    padded[0, :len(b)] = normalized
    index = {n: i for i, n in enumerate(model["names"])}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"Predicates absent from verified bank: {missing}. See RA ONNX Diagnostics for exact names.")
    rows = [index[n] for n in names]
    return {"image": np.ascontiguousarray(x), "boxes": padded,
            "box_counts": np.array([len(b)], dtype=np.int64),
            "W": np.ascontiguousarray(model["bank"]["W"][rows], dtype=np.float32),
            "alpha": np.ascontiguousarray(model["bank"]["alpha"][rows], dtype=np.float32)}


def decode_outputs(out, names, count, source_ids, calibration, threshold, topk, pair_weight):
    p, q, si, oi, valid = (out[n] for n in OUTPUT_SPEC)
    if p.ndim != 3 or p.shape[0] != 1 or p.shape[2] != len(names):
        raise ValueError(f"Unexpected pred_logits shape: {p.shape}")
    expected = p.shape[:2]
    if any(x.shape != expected for x in (q, si, oi, valid)):
        raise ValueError("Inconsistent output shapes.")
    if si.dtype != np.int64 or oi.dtype != np.int64 or valid.dtype != np.bool_:
        raise ValueError("Unexpected output index/mask dtypes.")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("ONNX returned nonfinite logits.")
    a, b = float(calibration["a"]), float(calibration["b"])
    z = a * (p[0] + float(pair_weight) * q[0, :, None]) + b
    score = np.empty_like(z)
    positive = z >= 0
    score[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    score[~positive] = ez / (1 + ez)
    sub, obj = si[0], oi[0]
    keep = valid[0] & (sub >= 0) & (obj >= 0) & (sub < count) & (obj < count) & (sub != obj)
    best = score.argmax(axis=1)  # Graph-constrained: one predicate per directed pair.
    candidates = [k for k in np.flatnonzero(keep) if score[k, best[k]] >= threshold]
    candidates.sort(key=lambda k: -float(score[k, best[k]]))
    result = []
    seen = set()
    for k in candidates:
        s, o, v = int(sub[k]), int(obj[k]), int(best[k])
        if (s, o) in seen:
            continue
        seen.add((s, o))
        result.append({"subject_idx": s, "object_idx": o, "subject_source_idx": int(source_ids[s]),
                       "object_source_idx": int(source_ids[o]), "predicate": names[v],
                       "score": float(score[k, v])})
        if len(result) >= topk:
            break
    return result


class RAOnnxPredictV06:
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("relations_text", "relations_json")
    FUNCTION = "predict"
    CATEGORY = "RelateAnything/ONNX v0.6"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ra_onnx_model": ("RA_ONNX_MODEL",), "image": ("IMAGE",),
                             "vocabulary": ("STRING", {"default": DEFAULT_VOCAB, "multiline": True}),
                             "topk": ("INT", {"default": 20, "min": 1, "max": 128}),
                             "score_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                             "pair_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0})},
                "optional": {"regions": ("RA_REGIONS",), "masks": ("MASK",),
                             "boxes_json": ("STRING", {"forceInput": True})}}

    def predict(self, ra_onnx_model, image, vocabulary, topk, score_threshold, pair_weight,
                regions=None, masks=None, boxes_json=None):
        model = _one(ra_onnx_model, "ra_onnx_model")
        if not model.get("ready"):
            raise RuntimeError("ONNX inference blocked: " + model.get("blocked_reason", "Unverified model")
                               + ". Run RA ONNX Diagnostics.")
        if sum(v is not None for v in (regions, masks, boxes_json)) != 1:
            raise ValueError("Connect exactly one region source: RA_REGIONS, masks, or boxes_json.")
        rgb = _image(image)
        if regions is None:
            regions, _ = collect_regions(image, masks, boxes_json)
        else:
            regions = _one(regions, "regions")
        boxes, ids = _validate_regions(regions, *rgb.shape[:2])
        if len(boxes) < 2:
            raise ValueError(f"At least two separate regions required; got {len(boxes)}. Check SAM3 output/debug.")
        vocab = str(_one(vocabulary, "vocabulary"))
        names = list(dict.fromkeys(s.strip() for s in vocab.replace(",", "\n").splitlines() if s.strip()))
        if not names:
            names = [str(n) for n in model["bank"]["default"]]
        k = int(_one(topk, "topk"))
        threshold = float(_one(score_threshold, "score_threshold"))
        weight = float(_one(pair_weight, "pair_weight"))
        if not 1 <= k <= 128 or not 0 <= threshold <= 1 or not 0 <= weight <= 3:
            raise ValueError("Invalid topk/threshold/pair_weight.")
        feed = make_feed(rgb, boxes, names, model)
        raw = model["session"].run(list(OUTPUT_SPEC), feed)
        results = decode_outputs(dict(zip(OUTPUT_SPEC, raw)), names, len(boxes), ids,
                                 model["meta"]["calibration"], threshold, k, weight)
        lines = [f"(region{r['subject_idx']} / SAM3 #{r['subject_source_idx']}) --{r['predicate']} "
                 f"[{r['score']:.4f}]--> (region{r['object_idx']} / SAM3 #{r['object_source_idx']})"
                 for r in results]
        text = "\n".join(lines) if lines else "No relations passed the selected threshold."
        payload = {"version": VERSION, "relations": results, "boxes_xyxy": boxes.tolist(),
                   "source_indices": list(ids), "vocabulary": names, "score_threshold": threshold,
                   "pair_weight": weight, "calibration": model["meta"]["calibration"],
                   "geometry": "boxes only; masks converted to extents", "model_path": model["path"]}
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        print("[RA ONNX v0.6]\n" + text)
        return {"ui": {"text": [text]}, "result": (text, data)}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (RARegionsV06, RAOnnxLoadV06, RAOnnxInspectV06, RAOnnxPredictV06)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RARegionsV06": "RA · Regions from SAM3 · v0.6",
    "RAOnnxLoadV06": "RA · Load Local ONNX · v0.6",
    "RAOnnxInspectV06": "RA · ONNX Diagnostics · v0.6",
    "RAOnnxPredictV06": "RA · ONNX Predict Relations · v0.6",
}
