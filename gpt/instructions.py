from functools import partial
import time 
import tiktoken 
import torch 
import json 
import os
import urllib
from torch.utils.data import DataLoader, Dataset
from configs import GPT_SMALL 
from trained_model import get_small_gpt_2_model
from train import train_model_simple, calc_loss_loader
from generate import generate, text_to_token_ids, token_ids_to_text
import matplotlib.pyplot as plt

INSTRUCTIONS_FILE_PATH="instruction-data.json"
INSTRUCTIONS_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
)

TRAINED_MODEL_CACHE = 'instructions.pth'

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
)
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

def partition_data():
    data = download_and_load_file(url=INSTRUCTIONS_URL, file_path=INSTRUCTIONS_FILE_PATH)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion 

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    return train_data, val_data, test_data

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def plot_values(
    epochs_seen, examples_seen, train_values, val_values, label="loss"
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, label=f"Validation {label}", linestyle="-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def get_fine_tuned_model(skip_cache=False):
    gpt = get_small_gpt_2_model()
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    gpt.to(device)
    
    cached = None
    if not skip_cache:
        try: 
            cached = torch.load(TRAINED_MODEL_CACHE)
        except:
            print("Unabled to find cached fine-tuned classifier model.")
        if cached:
            print("Found cached model, skipping classification fine tuning.")
            gpt.load_state_dict(cached)
            gpt.eval()

    train_data, val_data, test_data = partition_data()
    
    # training
    if not cached:
        num_workers = 0
        batch_size = 8
        tokenizer = tiktoken.get_encoding('gpt2')
        torch.manual_seed(123)
        
        customized_collate_fn = partial(
            custom_collate_fn,
            device=device,
            allowed_max_length=1024
        )

        train_dataset = InstructionDataset(train_data, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )
        val_dataset = InstructionDataset(val_data, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        test_dataset = InstructionDataset(test_data, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        start_time = time.time()
        optimizer = torch.optim.AdamW(
            gpt.parameters(), lr=0.00005, weight_decay=0.1
        )
        num_epochs = 2

        train_losses, val_losses, tokens_seen = train_model_simple(
            model=gpt, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=format_input(val_data[0]), tokenizer=tokenizer, device=device
        )

        # plot
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_values(epochs_tensor, tokens_seen, train_losses, val_losses)

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        # cache trained model
        if not skip_cache:
            torch.save(gpt.state_dict(), TRAINED_MODEL_CACHE)

    # testing
    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
                model=gpt,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=GPT_SMALL["context_length"],
                eos_id=50256
            )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    return gpt
