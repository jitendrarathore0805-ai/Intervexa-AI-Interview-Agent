
import logging
import os
import threading
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

MODEL_PATH = Path(
    os.getenv("QWEN_MODEL_PATH", "./models/qwen")
)

_tokenizer = None
_model = None
_device = None
_load_lock = threading.Lock()


def _load_model():
    """Load Qwen only when it is first needed."""
    global _tokenizer, _model, _device

    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model, _device

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model, _device

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Qwen model folder not found: {MODEL_PATH.resolve()}"
            )

        logger.info("Loading Qwen tokenizer from %s", MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading Qwen model on %s", device)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=dtype,
            local_files_only=True,
        )
        model.to(device)
        model.eval()

        _tokenizer = tokenizer
        _model = model
        _device = device

        logger.info("Qwen loaded successfully on %s", device)

    return _tokenizer, _model, _device


def generate_text(
    prompt,
    max_new_tokens=300,
    temperature=0.3,
    system_prompt=None,
):
    """Generate text using the local Qwen model."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    tokenizer, model, device = _load_model()

    if getattr(tokenizer, "chat_template", None):
        messages = []

        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt.strip(),
            })

        messages.append({
            "role": "user",
            "content": prompt.strip(),
        })

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(device)
    else:
        formatted_prompt = (
            f"{system_prompt.strip()}\n\n"
            if system_prompt and system_prompt.strip()
            else ""
        ) + prompt.strip()

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(device)

    generation_options = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": float(temperature) > 0,
        "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    }

    if float(temperature) > 0:
        generation_options["temperature"] = max(
            0.05, float(temperature)
        )
        generation_options["top_p"] = 0.9

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_options)

    # Decode generated tokens only, not the original prompt.
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_length:]

    result = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    if not result:
        raise RuntimeError("Qwen returned an empty response.")

    return result


def generate_question(role, skill, difficulty):
    """Compatibility wrapper for the earlier question generator."""
    prompt = f"""
Generate exactly one technical interview question.

Candidate role: {role}
Skill: {skill}
Difficulty: {difficulty}

Rules:
- Keep it relevant to the skill.
- Match the requested difficulty.
- Do not provide the answer.
- Return only the question.
"""
    return generate_text(
        prompt,
        max_new_tokens=120,
        temperature=0.3,
        system_prompt="You are a professional technical interviewer.",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    question = generate_question(
        "Machine Learning Engineer",
        "Random Forest",
        "Medium",
    )
    print("\nGenerated Interview Question:\n", question)
