from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = load_dataset("mnist", split="train[:5]")

for idx, sample in enumerate(dataset):
    image = sample["image"].convert("RGB")
    true_text = str(sample["label"])  # MNIST labels are digits

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Sample #{idx+1}")
    print("🖼️  Ground Truth :", true_text)
    print("🤖 Predicted    :", predicted_text)
    print("-" * 50)
