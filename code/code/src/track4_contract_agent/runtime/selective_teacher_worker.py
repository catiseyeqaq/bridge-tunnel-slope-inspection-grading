"""Private, one-run PEFT teacher process. Loading is paid inside the entry timer."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time


def main():
    started = time.perf_counter()
    request_path, output_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    cfg, digest = request["config"], request["request_sha256"]
    payload = {k: v for k, v in request.items() if k != "request_sha256"}
    assert hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == digest
    if output_path.exists():
        raise FileExistsError(output_path)
    if not request["jobs"]:
        return
    base, adapter = Path(cfg["base_model"]), Path(cfg["adapter"])
    for name,key in (('config.json','base_config_sha256'),('model.safetensors.index.json','base_index_sha256')):
        expected=cfg.get(key)
        if expected and hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Teacher foundation identity mismatch: '+name)
    for name, expected in (("adapter_model.safetensors", cfg["adapter_sha256"]),
                           ("adapter_config.json", cfg["adapter_config_sha256"])):
        with (adapter / name).open("rb") as fh:
            actual = hashlib.file_digest(fh, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Trained adapter identity mismatch: " + name)
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Unexpected model identity; Qwen3.8 is this project's directory alias")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel
    from PIL import Image, ImageOps
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 60 * 2**30:
        raise RuntimeError("Teacher needs at least 60 GiB free on the chosen device")
    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, trust_remote_code=True,
                                             min_pixels=256*28*28, max_pixels=1024*1024)
    model = AutoModelForImageTextToText.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, trust_remote_code=True, attn_implementation="sdpa",
        low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    model.eval()
    loaded = time.perf_counter()
    device = next(model.parameters()).device
    runtime = {"base_model": str(base), "model_type": config["model_type"],
               "adapter_sha256": cfg["adapter_sha256"], "load_s": loaded-started,
               "device": str(device), "torch": torch.__version__,
               "peak_host_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
               "peak_gpu_allocated_GiB": torch.cuda.max_memory_allocated()/2**30}
    output_path.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime), flush=True)

    class Deadline(StoppingCriteria):
        def __init__(self, deadline):
            self.deadline = deadline
        def __call__(self, input_ids, scores, **kwargs):
            return time.perf_counter() >= self.deadline

    with output_path.open("x", encoding="utf-8") as out:
        for index, job in enumerate(request["jobs"]):
            if time.perf_counter() - started >= cfg["wall_budget_s"] - 1:
                break
            begin = time.perf_counter()
            rec = {"row_index": job["row_index"], "request_sha256": digest}
            try:
                with Image.open(job["image_path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                             {"type": "text", "text": job["prompt"]}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
                visual = inputs.get("pixel_values")
                if visual is None or not visual.numel():
                    raise ValueError("Teacher image tensor is missing")
                image_tensor_sha = hashlib.sha256(visual.cpu().numpy().tobytes()).hexdigest()
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                image_budget = cfg.get("first_image_s", cfg["per_image_s"]) if index == 0 else cfg["per_image_s"]
                deadline = min(begin + image_budget, started + cfg["wall_budget_s"] - 0.5)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"],
                                               do_sample=False, use_cache=True,
                                               stopping_criteria=StoppingCriteriaList([Deadline(deadline)]))
                raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0]
                rec.update(ok=True, raw=raw, image_tensor_sha256=image_tensor_sha,
                           token_count=int(generated.shape[1] - inputs["input_ids"].shape[1]),
                           time_limit_reached=time.perf_counter() >= deadline)
                del inputs, generated
                image.close()
            except Exception as exc:
                rec.update(ok=False, error=type(exc).__name__ + ": " + str(exc)[:500])
            rec["elapsed_s"] = time.perf_counter() - begin
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({k: rec.get(k) for k in ("row_index", "ok", "elapsed_s", "token_count")}), flush=True)
    runtime["worker_s"] = time.perf_counter() - started
    runtime["peak_host_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    runtime["peak_gpu_allocated_GiB"] = torch.cuda.max_memory_allocated()/2**30
    output_path.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
