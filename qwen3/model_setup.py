import torch
from unsloth import FastVisionModel

def load_model_and_tokenizer():
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,  # Use 4-bit precision to reduce memory usage
        use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing for long context
    )

    # Configure LoRA adapters for efficient fine-tuning
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # Fine-tune vision layers
        finetune_language_layers=True,  # Fine-tune language layers
        finetune_attention_modules=True,  # Fine-tune attention layers
        finetune_mlp_modules=True,  # Fine-tune MLP layers
        r=16,  # Rank for LoRA
        lora_alpha=16,  # Alpha value for LoRA
        lora_dropout=0,  # Dropout rate for LoRA
        bias="none",
        random_state=3407,  # Random seed for reproducibility
        use_rslora=False,  # Disable rank-stabilized LoRA
        loftq_config=None,  # No LoftQ configuration
    )
    return model, tokenizer