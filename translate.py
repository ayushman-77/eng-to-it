import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from config import get_config, latest_weights_file_path, get_weights_file_path
from model import build_transformer, Transformer
from train import get_or_build_tokenizer, greedy_decode, load_opus_dataset

def load_inference_components(
    config: Optional[Dict[str, Any]] = None,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None
):
    cfg: Dict[str, Any] = get_config() if config is None else config
    dev: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    tokenizer_src_path = Path(str(cfg['tokenizer_file']).format(cfg['lang_src']))
    tokenizer_tgt_path = Path(str(cfg['tokenizer_file']).format(cfg['lang_tgt']))

    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        ds_raw = load_opus_dataset(str(cfg["lang_src"]), str(cfg["lang_tgt"]))
        tokenizer_src: Tokenizer = get_or_build_tokenizer(cfg, ds_raw, str(cfg['lang_src']))
        tokenizer_tgt: Tokenizer = get_or_build_tokenizer(cfg, ds_raw, str(cfg['lang_tgt']))
    else:
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    model: Transformer = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        int(cfg['seq_len']),
        int(cfg['seq_len']),
        d_model=int(cfg.get('d_model', 256)),
        N=int(cfg.get('N', 3)),
        h=int(cfg.get('h', 8)),
        dropout=float(cfg.get('dropout', 0.1)),
        d_ff=int(cfg.get('d_ff', 512))
    ).to(dev)

    if weights_path is None:
        weights_path = latest_weights_file_path(cfg)

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=dev)
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: No trained weights found. Model initialized with random weights.")

    model.eval()
    return cfg, model, tokenizer_src, tokenizer_tgt, dev

def translate(
    sentence: Union[str, int],
    config: Optional[Dict[str, Any]] = None,
    model: Optional[Transformer] = None,
    tokenizer_src: Optional[Tokenizer] = None,
    tokenizer_tgt: Optional[Tokenizer] = None,
    device: Optional[torch.device] = None,
    weights_path: Optional[str] = None
) -> str:
    text = str(sentence)
    
    if model is None or tokenizer_src is None or tokenizer_tgt is None or config is None:
        cfg, mdl, t_src, t_tgt, dev = load_inference_components(config, weights_path, device)
    else:
        cfg = config
        mdl = model
        t_src = tokenizer_src
        t_tgt = tokenizer_tgt
        dev = device if device is not None else next(model.parameters()).device

    sos_id = t_src.token_to_id('[SOS]')
    eos_id = t_src.token_to_id('[EOS]')
    pad_id = t_src.token_to_id('[PAD]')

    sos_idx = sos_id if sos_id is not None else 2
    eos_idx = eos_id if eos_id is not None else 3
    pad_idx = pad_id if pad_id is not None else 1

    sos_token = torch.tensor([sos_idx], dtype=torch.int64)
    eos_token = torch.tensor([eos_idx], dtype=torch.int64)
    pad_token = torch.tensor([pad_idx], dtype=torch.int64)

    enc_input_tokens = t_src.encode(text).ids
    enc_num_padding_tokens = int(cfg['seq_len']) - len(enc_input_tokens) - 2

    if enc_num_padding_tokens < 0:
        raise ValueError(f"Sentence is too long (max supported tokens: {int(cfg['seq_len']) - 2})")

    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ],
        dim=0
    ).unsqueeze(0).to(dev)

    encoder_mask = (encoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int().to(dev)

    model_out = greedy_decode(mdl, encoder_input, encoder_mask, t_src, t_tgt, int(cfg['seq_len']), dev)
    
    return str(t_tgt.decode(model_out.detach().cpu().numpy()))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        text_to_translate = " ".join(sys.argv[1:])
    else:
        text_to_translate = "I am a student and I love reading books."
    
    cfg = get_config()
    print(f"Input sentence: {text_to_translate}")
    output = translate(text_to_translate, config=cfg)
    print(f"Translation ({cfg['lang_tgt']}): {output}")
