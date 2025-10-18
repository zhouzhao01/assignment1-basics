import torch 
import torch.nn as nn
import torch.nn.functional as F
from  collections.abc import Callable, Iterable

class LLMDecoder:
    def __init__(self, temperature=1.0, top_p =0.9, max_length=256):
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length

    def apply_temperature(self, logits):
        return logits / self.temperature

    def apply_top_p(self, probs:torch.Tensor):
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

        accum_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = accum_probs > self.top_p

        if remove_mask.dim() > 0:
            remove_mask[..., 0] = False
        
        sorted_probs[remove_mask] = 0.0

        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(-1, indices, sorted_probs)
        # Renormalize after filtering
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
        return probs_filtered
    
    def sample(self, logits):
        logits = self.apply_temperature(logits)
        probs = F.softmax(logits, dim=-1)
        probs = self.apply_top_p(probs)

        next_token = torch.multinomial(probs[:, -1, :], num_samples=1)
        return next_token

    def generate(self, model, input_ids:torch.tensor, tokenizer, eos_token_id):
        print(tokenizer.decode(input_ids[0].tolist()))
        for _ in range(self.max_length):
            logits = model(input_ids)

            next_token = self.sample(logits)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            print(tokenizer.decode(next_token[0].tolist()))

            if next_token.item() == eos_token_id or input_ids.shape[-1] >= self.max_length:
                break
        
        return tokenizer.decode(input_ids[0].tolist())
    
def test_on_GPT2():
    """Test decoder on pretrained GPT-2 to verify sampling logic"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Wrapper to make HF model compatible with our decoder
    class ModelWrapper:
        def __init__(self, hf_model):
            self.model = hf_model

        def __call__(self, input_ids):
            return self.model(input_ids).logits

        def eval(self):
            self.model.eval()

    wrapped_model = ModelWrapper(model)

    test_text = "What is a computer?"
    tokens = tokenizer.encode(test_text, return_tensors='pt')

    decoder = LLMDecoder(temperature=0.8, top_p=0.9, max_length=50)
    output = decoder.generate(wrapped_model, tokens, tokenizer, tokenizer.eos_token_id)

    print("\n" + "="*50)
    print("Generated text:")
    print(output)
    print("="*50)

def test_on_custim_llm():
    from scaffoldings import load_checkpoint
    from basic_blocks import transformer_lm
    from optimizer import AdamW
    from dataset import plain_dataset
    import numpy as np

    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    vocab_size:int = 32000
    context_length:int = 128
    num_layers:int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

    epoch = 4
    batch_size = 8
    device = "cuda"

    data_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/owt_valid_encodings.npy"
    data = np.memmap(data_path,dtype=np.int32)
    plain_dataset_ins = plain_dataset(data,
                                      batch_size,
                                      context_length,
                                      device)

    tokens_per_batch = batch_size * context_length
    total_tokens = plain_dataset_ins.__len__()
    total_batches = total_tokens // tokens_per_batch

    tokens_per_batch = batch_size * context_length
    total_tokens = plain_dataset_ins.__len__()
    total_batches = total_tokens // tokens_per_batch

    model = transformer_lm(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        device
    )

    optimizer = AdamW(model.parameters(recurse=True),flag_lr_schedule=True,
                warmup_iters=int(total_batches*0.05),
                cosine_cycle_iters=total_batches)
 
    ckp_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/weights/1014/step_30000.pt"
    iterations = load_checkpoint(ckp_path, model, optimizer)

    test_text = "Bob has an apple, which is so red"
    
    # owt_tokenizer = Tokenizer(BPE()).from_file("/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/basic_blocks/tokenizer_owt.json")
    tiny_tokenizer = Tokenizer(BPE()).from_file("/basic_blocks/tokenizer_tiny_story.json")
    tokens = torch.tensor(tiny_tokenizer.encode(test_text).ids, device=device)
    tokens = torch.unsqueeze(tokens, dim=0)
    LLMDecoder_ins = LLMDecoder(temperature=0.8, top_p=0.9, max_length=50)
    text = LLMDecoder_ins.generate(model, tokens, tiny_tokenizer, 0)
    print(clean_bpe_text(text))

from scaffoldings import builder, load_checkpoint, set_random_seed
from tokenizers import Tokenizer
from tokenizers.models import BPE
def test_on_custom_llm():
    model_config_json_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/runs/transformer_lm_40m_4090/config.json"
    builder_ins = builder(model_config_json_path)
    set_random_seed(builder_ins.config["seed"])
    model = builder_ins.build_model()
    optimizer = builder_ins.build_optimizer(model)

    ckp_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/runs/transformer_lm_40m_4090/step_32655.pt"
    iterations = load_checkpoint(ckp_path,model,optimizer)

    test_text = "Bob has an apple, which is so red"

    tiny_tokenizer = Tokenizer(BPE()).from_file("basic_blocks/tokenizer_tiny_story.json")
    tokens = torch.tensor(tiny_tokenizer.encode(test_text).ids, device=builder_ins.config["device"])
    tokens = torch.unsqueeze(tokens, dim=0)
    LLMDecoder_ins = LLMDecoder(temperature=0.8, top_p=0.9, max_length=256)
    text = LLMDecoder_ins.generate(model, tokens, tiny_tokenizer, 0)
    print(clean_bpe_text(text))

def clean_bpe_text(text):
    """
    清理 BPE tokenizer 解码后的特殊符号
    
    常见符号：
    - Ġ: 空格
    - Ċ: 换行符
    - ĉ: 制表符
    """
    # 替换特殊符号
    replacements = {
        'Ġ': ' ',      # 空格
        'Ċ': '\n',     # 换行
        'ĉ': '\t',     # 制表符
        'âĢĻ': "'",    # 撇号
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

if __name__ == "__main__":
    # test_on_GPT2()
    # test_on_custim_llm()
    test_on_custom_llm()