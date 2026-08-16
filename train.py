from pathlib import Path
from typing import Dict, Any, Tuple, Iterator, Callable, Union, List
import os
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
from model import build_transformer, Transformer
from config import get_weights_file_path, latest_weights_file_path, get_config

def load_opus_dataset(src_lang: str, tgt_lang: str) -> Any:
    lang_pair = f"{src_lang}-{tgt_lang}"
    try:
        return load_dataset('Helsinki-NLP/opus_books', lang_pair, split='train')
    except Exception:
        return load_dataset('opus_books', lang_pair, split='train')

def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device
) -> torch.Tensor:
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    assert sos_idx is not None and eos_idx is not None

    # Recompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with the max probability (greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model: Transformer,
    validation_ds: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: Callable[[str], None],
    global_state: int,
    writer: Union[SummaryWriter, None],
    num_examples: int = 2
) -> None:
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Print to the console
            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                print_msg('-' * console_width)
                break

def get_all_sentences(ds: Any, lang: str) -> Iterator[str]:
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config: Dict[str, Any], ds: Any, lang: str) -> Tokenizer:
    tokenizer_path = Path(str(config['tokenizer_file']).format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    ds_raw = load_opus_dataset(str(config["lang_src"]), str(config["lang_tgt"]))

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, str(config["lang_src"]))
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, str(config["lang_tgt"]))

    # Filter out sentences that exceed max sequence length
    seq_limit = int(config["seq_len"]) - 2
    filtered_ds: List[Any] = []
    for item in ds_raw:
        src_len = len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids)
        tgt_len = len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids)
        if src_len <= seq_limit and tgt_len <= seq_limit:
            filtered_ds.append(item)

    print(f'Total sentences: {len(ds_raw)}, filtered within seq_len ({config["seq_len"]}): {len(filtered_ds)}')

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(filtered_ds))
    val_ds_size = len(filtered_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(filtered_ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, str(config["lang_src"]), str(config["lang_tgt"]), int(config["seq_len"]))
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, str(config["lang_src"]), str(config["lang_tgt"]), int(config["seq_len"]))

    train_dataloader = DataLoader(train_ds, batch_size=int(config['batch_size']), shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config: Dict[str, Any], vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        int(config['seq_len']), 
        int(config['seq_len']), 
        d_model=int(config.get('d_model', 256)),
        N=int(config.get('N', 3)),
        h=int(config.get('h', 8)),
        dropout=float(config.get('dropout', 0.1)),
        d_ff=int(config.get('d_ff', 512))
    )
    return model

def train_model(config: Dict[str, Any]) -> None:
    # Set optimal CPU thread count or GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    print(f'Using device: {device}')

    Path(str(config['model_folder'])).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(str(config['experiment_name']))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), eps=1e-9)

    initial_epoch = 0
    global_step = 0
    
    if config.get('preload') == 'latest':
        model_filename = latest_weights_file_path(config)
    elif config.get('preload'):
        model_filename = get_weights_file_path(config, str(config['preload']))
    else:
        model_filename = None

    if model_filename is not None and Path(model_filename).exists():
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    pad_idx = tokenizer_tgt.token_to_id('[PAD]')
    assert pad_idx is not None
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, int(config['num_epochs'])):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (Batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (Batch, seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', float(loss.item()), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, int(config['seq_len']), device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        if model_filename:
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    cfg = get_config()
    train_model(cfg)