import os
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Configuration
MODEL_ID = "openai/whisper-tiny"  # Change to small or base for better results
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "til26_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "whisper-finetuned")

print(f"CUDA Available: {torch.cuda.is_available()}")

# Assume we load a local dataset downloaded from the Drive API
try:
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
except Exception as e:
    print(f"Could not load local dataset. Make sure audio files are in {DATASET_PATH}.")
    print(f"Error: {e}")
    # For testing without the dataset, we exit. 
    # sys.exit(1)

processor = WhisperProcessor.from_pretrained(MODEL_ID, task="transcribe")

def preprocess_function(examples):
    audio = examples["audio"]
    # Feature extraction
    features = processor.feature_extractor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="np"
    )
    # Tokenization
    labels = processor.tokenizer(
        examples["text"],
        return_tensors="np"
    ).input_ids[0]

    return {
        "input_features": features.input_features[0],
        "labels": labels
    }

# Ensure models dir exists
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

print("Preprocessing dataset...")
# In a real run, you'd apply the map function to the dataset:
# processed_dataset = dataset.map(preprocess_function)

print("Setting up Trainer...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    max_steps=500,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    generation_max_length=225,
    report_to="none",
)

# Load model
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.use_cache = False

# We use a dummy trainer instantiation for structural completeness.
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=processed_dataset["train"],
#     tokenizer=processor.feature_extractor,
# )

print("Training script fully structured. Run trainer.train() when dataset is confirmed.")
# trainer.train()
# trainer.save_model(OUTPUT_DIR)
# processor.save_pretrained(OUTPUT_DIR)
