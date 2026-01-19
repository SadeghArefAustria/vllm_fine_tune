from transformers import TextStreamer

def run_inference(model, tokenizer, dataset):
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)  # Enable for inference

    # Example inference
    instruction = "Write the LaTeX representation for this image."
    image = dataset[2]["image"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Generate output
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1
    )