from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import evaluate

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = load_dataset("scene_text", split="test[:5]")
wer_metric = evaluate.load("wer")

references = []
predictions = []

for sample in dataset:
    image = sample["image"].convert("RGB")
    true_text = sample["text"]

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    references.append(true_text.lower())
    predictions.append(predicted_text.lower())

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"🔍 Word Error Rate (WER): {wer:.2f}")
