from datasets import load_dataset

# Moved the `instruction` variable outside the function to fix the scope issue
instruction = "Write the LaTeX representation for this image."

def load_and_process_dataset():
    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

    # Save LaTeX representation to a file
    latex = dataset[2]["text"]
    with open("latex_output.txt", "w") as f:
        f.write(latex)

    # Function to convert dataset samples to conversation format
    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["text"]},
                    ],
                },
            ]
        }

    # Convert the dataset to the required format
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    return dataset, converted_dataset