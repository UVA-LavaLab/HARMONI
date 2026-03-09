from itertools import product
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.model_config import make_model_config

def categorize_and_write_lines(lines, filename):
    with open(filename, "a") as f:
        for line in lines:
            f.write(line + "\n")

def generate_experiment_configurations():
    precisions = ["BF16"]

    #E2E runs
    # models = ['MISTRAL-7B', 'LLAMA2-7B']
    # batches = [1, 4, 8]
    # inputs = [128, 2048]
    # outputs = [128, 2048]
    # drams = ['DDR5-M4-R4-C8-8-A2', 'DDR5-M8-R4-C8-8-A2']

    # models = ['LLAMA3-70B']
    # batches = [1, 4, 8]
    # inputs = [128, 2048]
    # outputs = [128, 2048]
    # drams = ['DDR5-M16-R4-C8-8-A2', 'DDR5-M16-R8-C8-8-A2']  

    # #Scale out 
    # models = ['LLAMA2-7B']
    # batches = [1, 4, 8]
    # inputs = [128, 2048]
    # outputs = [128, 2048]
    # drams = ['DDR5-M1-R4-C8-8-A2', 'DDR5-M2-R4-C8-8-A2'] #'DDR5-M4-R4-C8-8-A2', 'DDR5-M8-R4-C8-8-A2'

    # #Scale up (non equal capacity) 
    # models = ['LLAMA2-7B']
    # batches = [1, 4, 8]
    # inputs = [2048]
    # outputs = [2048]
    # drams = ['DDR5-M4-R2-C4-8-A2', 'DDR5-M4-R2-C8-8-A2', 'DDR5-M4-R2-C16-8-A2',
    #          'DDR5-M4-R4-C4-8-A2', 'DDR5-M4-R4-C16-8-A2',
    #          'DDR5-M4-R8-C4-8-A2', 'DDR5-M4-R8-C8-8-A2', 'DDR5-M4-R8-C16-8-A2'] #'DDR5-M4-R4-C8-8-A2'
    
    #Memory stds study
    models = ['LLAMA2-7B', 'MISTRAL-7B']
    batches = [1, 4, 8]
    inputs = [128, 2048]
    outputs = [128, 2048]
    drams = ['DDR4-M8-R4-C8-8-A2', 'GDDR6-M8-R4-C8-8-A2'] #'DDR5-M8-R4-C8-8-A2'
    
    
    #params = []
    small_params = []
    mid_params = []
    large_params = []
    large1_params = []
    large2_params = []
    strict_large2_cases = {
        ('LLAMA2-7B', 'BF16', 'DDR5-M4-R8-C16-8-A2', 2048, 2048, 8),
    }
    strict_large1_cases = {
        ('LLAMA2-7B', 'BF16', 'DDR5-M4-R4-C8-8-A2', 128, 2048, 8),
        ('LLAMA2-7B', 'BF16', 'GDDR6-M8-R4-C8-8-A2', 128, 2048, 8),
        ('LLAMA2-7B', 'BF16', 'DDR4-M8-R4-C8-8-A2', 128, 2048, 8),
        ('LLAMA2-7B', 'BF16', 'DDR5-M8-R4-C8-8-A2', 128, 2048, 8),
    }
    strict_large_cases = {
        ('MISTRAL-7B', 'BF16', 'GDDR6-M8-R4-C8-8-A2', 128, 2048, 8),
        ('MISTRAL-7B', 'BF16', 'GDDR6-M8-R4-C8-8-A2', 2048, 2048, 8),
        ('MISTRAL-7B', 'BF16', 'DDR5-M8-R4-C8-8-A2', 2048, 2048, 8),
        ('MISTRAL-7B', 'BF16', 'DDR4-M8-R4-C8-8-A2', 2048, 2048, 8),
    }

    for model, precision, dram, input_len, output_len, batch in product(
            models, precisions, drams, inputs, outputs, batches):
        
        context_length = make_model_config(model,precision)["context_len"]
        if input_len + output_len <= context_length:
            line = f"{model} {precision} {dram} {input_len} {output_len} {batch}"
            case_key = (model, precision, dram, input_len, output_len, batch)

            #params.append(line)
            if case_key in strict_large2_cases:
                large2_params.append(line)
            elif case_key in strict_large1_cases:
                large1_params.append(line)
            elif case_key in strict_large_cases:
                large_params.append(line)
            elif model in ['LLAMA2-7B', 'LLAMA3-70B'] and input_len == 2048 and output_len == 2048 and batch > 1:
                large1_params.append(line)
            elif model == 'LLAMA3-70B':
                large_params.append(line)
            elif model == 'LLAMA2-7B' and input_len == 2048 and output_len == 2048 and batch == 1:
                large_params.append(line)
            elif model == 'LLAMA2-7B' and input_len == 128 and output_len == 2048 and batch > 1:
                large_params.append(line)
            elif model == 'LLAMA2-7B' and input_len == 128 and output_len == 128 and batch > 1:
                mid_params.append(line)
            elif model == 'LLAMA2-7B' and input_len == 128 and output_len == 2048 and batch == 1:
                mid_params.append(line)
            elif model == 'MISTRAL-7B' and input_len == 2048 and output_len == 2048:
                mid_params.append(line)
            elif model == 'MISTRAL-7B' and input_len == 128 and output_len == 2048 and batch > 1:
                mid_params.append(line)
            else:
                small_params.append(line)



    #categorize_and_write_lines(params, "params/ispass_ae.txt")
    categorize_and_write_lines(small_params, "params/ispass_ae_small.txt")
    categorize_and_write_lines(mid_params, "params/ispass_ae_mid.txt")
    categorize_and_write_lines(large_params, "params/ispass_ae_large.txt")
    categorize_and_write_lines(large1_params, "params/ispass_ae_large1.txt")
    categorize_and_write_lines(large2_params, "params/ispass_ae_large2.txt")

    print("Experiment configurations written to params/.")

if __name__ == "__main__":
    generate_experiment_configurations()
