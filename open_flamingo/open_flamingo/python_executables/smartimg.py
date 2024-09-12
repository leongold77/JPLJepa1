from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)
# grab model checkpoint from huggingface hub-3b
from huggingface_hub import hf_hub_download
import torch
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

from PIL import Image
import requests
import torch
if torch.cuda.is_available():
    print("CUDA is available! Using GPU for calculations.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU for calculations.")
    device = torch.device("cpu")
model.to(device)

#example 1
import matplotlib.pyplot as plt
image_1 = Image.open("/home/leon-gold/Downloads/Apollo11_Flag_Wall.jpg")
# Display the query image with a larger size
plt.figure(figsize=(8, 8))  # Set the figure size to 8x8 inches
plt.imshow(image_1)
plt.axis('off')  # Hide the axis
plt.show()


"""
Step 2: Preprocessing images
"""
vision_x = [image_processor(image_1).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ['<image>The subject of this image is from '],
    return_tensors="pt",
     )
"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x = vision_x.to(device),
    lang_x = lang_x["input_ids"].to(device),
    attention_mask = lang_x["attention_mask"].to(device),
    max_new_tokens=50,
    num_beams=3,
)
print("Generated text: ", tokenizer.decode(generated_text[0]))