python3 run.py --model_name LLAMA2-7B --dtype BF16 --dram DDR5-M4-R4-C8-8-A2
 -i 32 -o 16 -b 1 --simulate --optimization layer static_mapping --fused_attn --fused_qkv 
