from misc.type import DataType

def make_model_config(name, dtype):
    
    model_table = {}
    model_table['GPT-175B'] = [96, 12288, 96, 96, 128, 4, 49152, 1, 4096, 32000, 'gelu']
    model_table['LLAMA2-7B'] = [32, 4096, 32, 32, 128, 8 / 3, 11008, 1, 4096, 32000, 'silu']
    model_table['LLAMA2-13B'] = [40, 5120, 40, 40, 128, 4, 20480, 1, 4096, 32000, 'silu']
    model_table['LLAMA3-70B'] = [80, 8192, 64, 8, 128, 7 / 2, 28672, 1, 8192, 128256, 'silu'] #https://huggingface.co/meta-llama/Meta-Llama-3-70B
    model_table['MISTRAL-7B'] = [32, 4096, 32, 8, 128, 7/2, 14336, 1, 8192, 32000, 'silu'] #https://huggingface.co/mistralai/Mistral-7B-v0.1
    model_table['PHI2-2B'] = [32, 2560, 32, 32, 80, 4, 10240, 1, 2048, 51200, 'gelu_new'] #https://huggingface.co/microsoft/phi-2/blob/main/config.json   (2.7B)
    model_table['PHI3-4B'] = [32, 3072, 32, 32, 96, 8/3, 8192, 1, 4096, 32064, 'silu'] #http://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json (phi3 mini, 3.8B)
    model_table['PHI3-7B'] = [32, 4096, 32, 8, 128, 7/2, 14336, 1, 8192, 100352, 'gegelu' ] #https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/config.json (phi3 small 8k variant)

    #Get model parameters or raise an error if the model is not found
    if name not in model_table:
        raise ValueError(f"Model {name} not found in model_table")

    nlayers, hdim, nheads, kvheads, dhead, ff_scale, intmdt_size, gqa_size, context_len, vocab, act_fn = model_table[name]
    config = {
        'name': name,
        'nlayers': nlayers,
        'hdim': hdim,
        'head_dim': hdim//nheads,
        'num_heads': nheads,
        'kv_heads': kvheads,
        'dhead': dhead, 
        'ff_scale': ff_scale, #CLEANUP: remove ff_scale (not used)
        'intmdt_size': intmdt_size,
        'gqa_size': gqa_size, #CLEANUP: remove gqa_size (not used anywhere)
        'context_len': context_len,
        'vocab': vocab,
        'dtype': dtype,
        'act_fn': act_fn
    }
    return config 