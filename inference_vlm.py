
#  # baseline eval
# baseline_llm_name = "Qwen/Qwen2.5-VL-3B-Instruct" # As requested
# print(f"--- Loading Baseline LLM: {baseline_llm_name} ---")
# try:
#     # Load with similar settings as the gaze base model for fair comparison
#     llm_base = AutoModelForCausalLM.from_pretrained(
#         baseline_llm_name,
#         device_map="auto", # Use auto device map
#         torch_dtype=llm_torch_dtype, # Use same dtype if possible
#         trust_remote_code=True
#     )
#     llm_base.eval()
#     print(f"Baseline LLM loaded on device(s): {llm_base.device}")
#     baseline_llm_device = llm_base.device
# except Exception as e:
#     print(f"ERROR: Failed to load baseline model '{baseline_llm_name}': {e}")
#     print("Baseline comparison will be skipped.")
#     llm_base = None # Set to None if loading fails



from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
import torch
from PIL import Image
import base64
import io
import os # For file path operations and creating a dummy file

def load_image_as_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            # Read the image bytes
            image_bytes = image_file.read()
            # Encode the bytes to a base64 string
            base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            print(f"Image from {image_path} loaded and base64 encoded.")
            return base64_encoded_image
    except Exception as e:
        print(f"Error loading or encoding image from {image_path}: {e}")
        return None

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
                files.append(entry)
    
    return files

def generate_text_with_qwen_vl(prompt: str, image_list: list[str] = None, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    print(f"Loading processor and model for: {model_name}...")
    try:
        # Load the processor, which combines the tokenizer and image processor for Qwen-VL.
        processor = AutoProcessor.from_pretrained(model_name)

        # Load the model. AutoModelForCausalLM is used for text generation.
        # We specify bfloat16 for improved performance on compatible hardware and automatically map to available devices.
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",trust_remote_code=True
        )
        print("Model and processor loaded successfully.")

      
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
        inputs = processor(messages, return_tensors="pt").to(model.device)

        print("Generating text...")
        # Generate text using the model. The 'inputs' dictionary contains all necessary components.
        # `max_new_tokens` controls the length of the generated response.
        # `temperature` influences the creativity/randomness.
        # `do_sample=True` enables sampling-based generation.
        # `pad_token_id` is crucial for handling variable-length inputs correctly during generation.
        output = model.generate(
            **inputs, # Unpack the dictionary of inputs (input_ids, attention_mask, pixel_values)
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id # Use the tokenizer from the processor
        )
        print("Text generation complete.")

        # Decode only the newly generated tokens from the output.
        # The `output` tensor includes the input tokens, so we slice to get just the new ones.
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(new_tokens, skip_special_tokens=True)

        return response_text.strip()

    except ImportError:
        return "Error: The 'transformers', 'torch', or 'Pillow' libraries are not installed. Please install them using 'pip install transformers torch Pillow'."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    folder = "training_data_heatmap/jsons_new"
    text_folder = "text_files/txts"
    subfolders = find_subfolders(folder)
    print(f"Subfolders in '{folder}': {subfolders}")
    for number in subfolders:
        image_folder = os.path.join(folder, number)
        text_file = os.path.join(text_folder, f"tofel_{number}.txt")
        png_files = sorted(find_png(image_folder))
        print(f"Image folder: {image_folder}, Text file: {text_file}, PNG files: {png_files}")



        text_prompt = "Provide a brief summary of the history of artificial intelligence, focusing on key milestones."
        response_text_only = generate_text_with_qwen_vl(text_prompt, image_list=png_files)
        print("Generated Response (Text Only):")
        print(response_text_only)
