#!/usr/bin/env python3
"""
查看 Pythia 70M 模型 tokenizer 的 token 数量
"""

from transformers import AutoTokenizer

def check_tokenizer_vocab_size():
    """获取 Pythia 70M tokenizer 的 vocabulary size"""
    
    print("=" * 60)
    print("Pythia 70M Tokenizer Information")
    print("=" * 60)
    
    # 方式1: 从 Hugging Face Hub 加载 tokenizer
    print("\n[方式1] 从 HuggingFace Hub 加载 (需要网络):")
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
        vocab_size = len(tokenizer)
        print(f"✓ Tokenizer vocab size: {vocab_size:,}")
        print(f"✓ Tokenizer type: {type(tokenizer).__name__}")
        
        # 额外信息
        if hasattr(tokenizer, 'vocab_size'):
            print(f"✓ tokenizer.vocab_size: {tokenizer.vocab_size:,}")
        
        # 显示一些特殊 token
        print(f"\n特殊 tokens:")
        print(f"  - BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        print(f"  - EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"  - PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"  - UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    # 方式2: 从本地 tokenizer 文件加载
check_tokenizer_vocab_size()