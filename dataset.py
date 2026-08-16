from typing import Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):

    def __init__(self, ds: Any, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        sos_id = tokenizer_src.token_to_id('[SOS]')
        eos_id = tokenizer_src.token_to_id('[EOS]')
        pad_id = tokenizer_src.token_to_id('[PAD]')

        self.sos_token = torch.tensor([sos_id if sos_id is not None else 2], dtype=torch.int64)
        self.eos_token = torch.tensor([eos_id if eos_id is not None else 3], dtype=torch.int64)
        self.pad_token = torch.tensor([pad_id if pad_id is not None else 1], dtype=torch.int64)
        
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Dict[str, Any]:
        src_target_pair = self.ds[index]
        src_text = str(src_target_pair['translation'][self.src_lang])
        tgt_text = str(src_target_pair['translation'][self.tgt_lang])

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return (mask == 0).type(torch.int)