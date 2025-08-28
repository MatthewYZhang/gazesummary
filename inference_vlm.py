from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import torch
import os  # For file path operations and creating a dummy file


def find_subfolders(folder_path):
    subfolders = []

    if os.path.isdir(folder_path):
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                subfolders.append(entry)

    return subfolders


def find_png(folder_path: str, suffix="png"):
    files = []

    if os.path.isdir(folder_path):
        for entry in os.listdir(folder_path):
            if entry.lower().endswith(f".{suffix.lower()}"):
                files.append(os.path.join(folder_path, entry))

    return files


def generate_text_with_qwen_vl(processor, model, prompt: str, image_list: list[str] = None, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    print(f"Loading processor and model for: {model_name}...")
    try:
        # Prepare chat messages in the format expected by Qwen-VL for multimodal input.
        # This structure allows combining an image and a text prompt.
        if image_list:
            image_list = [{"type": "image", "image": i} for i in image_list]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *image_list,
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            # If no image is provided, simply use the text prompt.
            messages = [
                {"role": "user", "content": prompt}
            ]

        # Process the messages to get model inputs (input_ids, attention_mask, pixel_values).
        # `return_tensors="pt"` ensures PyTorch tensors are returned.
        # `to(model.device)` moves the input tensors to the same device as the model.
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = inputs = processor(
            text=[text],
            images=image_inputs,
            # videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        print("Generating text...")
        # Generate text using the model. The 'inputs' dictionary contains all necessary components.
        # `max_new_tokens` controls the length of the generated response.
        # `temperature` influences the creativity/randomness.
        # `do_sample=True` enables sampling-based generation.
        # `pad_token_id` is crucial for handling variable-length inputs correctly during generation.
        output = model.generate(
            # Unpack the dictionary of inputs (input_ids, attention_mask, pixel_values)
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            # Use the tokenizer from the processor
            pad_token_id=processor.tokenizer.eos_token_id
        )
        print("Text generation complete.")

        # Decode only the newly generated tokens from the output.
        # The `output` tensor includes the input tokens, so we slice to get just the new ones.
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(new_tokens, skip_special_tokens=True)

        return response_text.strip()

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    print("Model and processor loaded successfully.")

    folder = "person_training_data_heatmap"
    subfolders = find_subfolders(folder)
    print(f"Subfolders in '{folder}': {subfolders}")
    for subfolder in subfolders:
        image_folder = os.path.join(folder, subfolder)
        text_file = os.path.join(folder, subfolder, "article.txt")
        png_files = sorted(find_png(image_folder))
        f = open(text_file, 'r')
        text = f.read()
        print(
            f"Image folder: {image_folder}, Text file: {text_file}, PNG files: {png_files}")
        text_prompt = f"Generate a one-paragraph 150-word personalized summary for the following article based on user's gaze heatmap. Bright and warm colors like red or orange mean the user spends more time on it, and dim and cold colors like blue mean the user spends less time on it. A good personalized summary should include more contents that the user spend more time reading and touch other contents briefly. For the content that the user is focused on, a better personalized summary should contain more details and be more consistent with the original statements. More generally, a good summary should be more comprehensive and of better quality. For comprehensiveness, while covering as much of the user's focused topic as possible, a good summary should also touch on other aspects. For quality, please consider four aspects: (1) Consistency - the factual alignment between the summary and the summarized source. (2) Coherence - the collective quality of all sentences. The summary should be well-structured and well-organized. (3) Relevance - selection of important content from the source. (4) Fluency - the quality of individual sentences. Based on this heatmap and text, you should first identify which sentences and paragraphs the user spends more time reading and then generate a personalized summary of this article based on these contents. Do not explain or say any of your analysis."
        response_text_only = generate_text_with_qwen_vl(
            processor, model, text_prompt + "\n\nArticle Text:\n" + text, image_list=png_files)
        print("Generated Response (Text Only):")
        print(response_text_only)
