import torch
import time

def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens:int = 50):

    inputs = tokenizer(prompt, return_tensor = 'pt')
     # Move inputs only if model is on a known torch device
    # try:
    #     device = next(model.parameters()).device
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
    # except StopIteration:
    #     pass
    # except Exception:
    #     pass

    print("DEBUG prompt:", prompt)
    print("DEBUG inputs type:", type(inputs))
    print("DEBUG input_ids type:", type(inputs["input_ids"]))
    print("DEBUG attention_mask type:", type(inputs.get("attention_mask")))
    print("DEBUG input_ids:", inputs["input_ids"])

    input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long, device=model.device)
    attention_mask = torch.tensor([inputs["attention_mask"]], dtype=torch.long, device=model.device)

    # if torch.cuda.is_available():
    #     inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()

    # Under the hood it performs:
    # prefill on the full prompt
    # then decode loop one token at a time
    # uses KV cache internally (usually, if enabled by model/config)
    # stops when:
    # it reaches max_new_tokens
    # or it hits EOS token (if generated)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    total_time = time.time() - start

    output_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

    input_len = input_ids.shape[1]
    output_len = outputs.shape[1]
    # generate() returns the full sequence: prompt tokens + newly generated tokens
    new_tokens =  output_len - input_len
    tokens_per_sec = new_tokens/total_time if total_time > 0 else 0.0

    return {
        "prompt": prompt, 
        "generated_text": output_text, 
        "latency_in_sec": total_time,
        "new_tokens": new_tokens, 
        "tokens_per_sec": tokens_per_sec 
    }
