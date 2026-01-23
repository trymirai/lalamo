import sys
from pathlib import Path


repo_root = str(Path(__file__).resolve().parent)
if repo_root in sys.path:
    sys.path.remove(repo_root)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main() -> None:
    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("content:", content)

    keys = sorted(model.state_dict().keys())
    print(f"\nstate_dict keys ({len(keys)}):")
    for key in keys:
        print(key)


if __name__ == "__main__":
    main()
