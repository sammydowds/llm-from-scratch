from configs import GPT_CONFIG_124M 
from loaders import create_dataloader_v1
from gpt import GPTModel
from train import train_model_simple
import tiktoken
import torch

SIMPLY_TRAINED_MODEL_CACHE_PATH = "simply-trained-model.pth"

def get_simply_trained_model(file_path = "the-verdict.txt", skip_cache = False):
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding('gpt2')

    # check if cached 
    if not skip_cache:
        existing = None
        try: 
            existing = torch.load(SIMPLY_TRAINED_MODEL_CACHE_PATH)
        except:
            print("Unabled to find cached model.")
        if existing:
            print("Found cached model, skipping training.")
            model.load_state_dict(existing)
            return model, { "meta": {}, "tokenizer": tokenizer}
    else:
        print("Skipping cache read. Initializing training...")

    with open(file_path, "r", encoding='utf-8') as file:
        text_data = file.read()
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # cache model
    if not skip_cache:
        print("Training complete. Caching model.")
        torch.save(model.state_dict(), SIMPLY_TRAINED_MODEL_CACHE_PATH)

    return model, { "meta": { "train_losses": train_losses, "val_losses": val_losses, "tokens_seen": tokens_seen, }, "tokenizer": tokenizer }  
