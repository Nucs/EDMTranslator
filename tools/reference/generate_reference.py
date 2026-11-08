"""Regenerate the reference translations the parity tests compare against (src/EDMTranslatorTest/reference_translations.json).

Usage:
    pip install torch transformers
    python tools/reference/generate_reference.py [path/to/reference_translations.json] [--check]

For every model and row of the file, this runs transformers' own greedy search - EncoderDecoderModel.generate with
num_beams=1, do_sample=False and no logits processor - on the model's PyTorch weights from the Hugging Face hub (CPU,
float32), fed the row's input_ids (what EDMTranslator's BertJa2GPTTokenizer produces for the sentence), and writes the
generated ids without the decoder start token into output_ids: the exact sequence EDMTranslator's greedy search must hand
to Tokenizer.Decode. max_length is max_new_tokens + 1, because transformers counts the decoder start token in it.

With --check the file is not written; the script exits 1 if any stored output_ids differ from a fresh run.
"""
import json
import re
import sys

import torch
from transformers import EncoderDecoderModel

torch.set_grad_enabled(False)


def generate(model: EncoderDecoderModel, spec: dict, input_ids: list) -> list:
    """Greedy-decode one sentence the way EDMTranslator does and return the new tokens.

    Args:
        model: The loaded EncoderDecoderModel (evaluation mode, float32).
        spec: The model's entry of the reference file (decoder_start_token_id, eos_token_id, max_new_tokens).
        input_ids: The encoder input ids of the sentence.

    Returns:
        The generated token ids without the decoder start token, EOS included when the search reached it.

    Raises:
        AssertionError: generate() did not start from the decoder start token.
    """
    ids = torch.tensor([input_ids], dtype=torch.long)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_length=spec["max_new_tokens"] + 1,
        num_beams=1,
        do_sample=False,
        decoder_start_token_id=spec["decoder_start_token_id"],
        bos_token_id=spec["decoder_start_token_id"],
        eos_token_id=spec["eos_token_id"],
        pad_token_id=spec["eos_token_id"],
        use_cache=True,
    )[0].tolist()
    assert out[0] == spec["decoder_start_token_id"], f"generate() did not start with the decoder start id: {out[:3]}"
    return out[1:]


def main() -> int:
    """Refresh (or with --check, compare) every row's output_ids.

    Returns:
        0 on success; 1 when --check found a row whose stored ids differ from a fresh run.
    """
    args = [a for a in sys.argv[1:] if a != "--check"]
    check = "--check" in sys.argv[1:]
    path = args[0] if args else "src/EDMTranslatorTest/reference_translations.json"
    data = json.load(open(path, encoding="utf-8"))
    differences = 0
    for key, spec in data["models"].items():
        model = EncoderDecoderModel.from_pretrained(spec["hub"]).eval().to(torch.float32)
        for row in spec["rows"]:
            fresh = generate(model, spec, row["input_ids"])
            if fresh != row.get("output_ids"):
                differences += 1
                print(f"{key}: {row['sentence']}: {row.get('output_ids')} -> {fresh}")
            row["output_ids"] = fresh
        del model
    if check:
        print(f"{differences} row(s) differ")
        return 1 if differences else 0
    text = json.dumps(data, ensure_ascii=False, indent=2)
    # One line per id list: the ids read as sequences.
    text = re.sub(r"\[\s*((?:-?\d+\s*,\s*)*-?\d+)\s*\]",
                  lambda m: "[" + ", ".join(re.split(r"\s*,\s*", m.group(1).strip())) + "]", text)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path} ({differences} row(s) changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
