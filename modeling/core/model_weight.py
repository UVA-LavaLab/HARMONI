from collections import OrderedDict
from misc.type import *
from modeling.core.tensor import *
from args import *

class ModelWeightInfo:
    def __init__(self, model):
        """
        Initializes weight metadata for profiling.
        """

        self.model = {
            "name": model["name"],
            "nlayers": model["nlayers"],
            "hdim": model["hdim"],
            "num_heads": model["num_heads"],
            "kv_heads": model["kv_heads"],
            "dhead": model["dhead"],
            "ff-scale": model["ff_scale"],
            "intmdt_size": model["intmdt_size"],
            "qga_size": model["gqa_size"],
            "context": model["context_len"],
            "vocab": model["vocab"],
            "dtype": model["dtype"],
            "head_dim": model["hdim"]//model["num_heads"] 
        }

        # NOTE: Can be updated if need be for different mapping schemes
        self.weights_new = OrderedDict()
        from args import get_args
        args = get_args()
        for layer in range(model["nlayers"]):
            if args.fused_qkv != 1:
                self.weights_new[f"layer_{layer}"] = {                
                    "Wq": OrderedDict({
                        h : HarmoniTensor(name="t_Wq", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['head_dim']))
                        for h in range(model["num_heads"])                                           
                    }),
                    "Wk": OrderedDict({
                        h : HarmoniTensor(name="t_Wk", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['head_dim']))
                        for h in range(model["kv_heads"])                                           
                    }),
                    "Wv": OrderedDict({
                        h : HarmoniTensor(name="t_Wv", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['head_dim']))
                        for h in range(model["kv_heads"])                                           
                    }),
                }
            else:
                self.weights_new[f"layer_{layer}"] = {
                    "Wqkv": HarmoniTensor(name="t_Wqkv", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], (model['num_heads']+model['kv_heads']+model['kv_heads'])*model['head_dim'])), 
                }
            self.weights_new[f"layer_{layer}"].update({
                "Wo": HarmoniTensor(name="t_Woproj", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['hdim'])),
            })
            if "LLAMA" in model["name"] or "MISTRAL" in model["name"]: #NOTE: currently only chekcing GQA, so everything else is same
                self.weights_new[f"layer_{layer}"].update({
                    "Wup": HarmoniTensor(name="t_Wup", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['intmdt_size'])),
                    "Wgate": HarmoniTensor(name="t_Wgate", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['intmdt_size'])),
                    "Wdown": HarmoniTensor(name="t_Wdown", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['intmdt_size'], model['hdim'])),
                })
            elif "GPT" in model["name"]:
                self.weights_new[f"layer_{layer}"].update({
                    "Wff1": HarmoniTensor(name="t_Wff1", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['intmdt_size'])),
                    "Wff2": HarmoniTensor(name="t_Wff2", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['intmdt_size'], model['hdim'])),
                })
            self.weights_new[f"layer_{layer}"].update({
                "rms_norm1": HarmoniTensor(name="t_rmsnorm1", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'],)),
                "rms_norm2": HarmoniTensor(name="t_rmsnorm2", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'],)),
            })
        self.weights_new["W_embed"] = HarmoniTensor(name="t_Wemb", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['vocab'], model['hdim']))
        self.weights_new["W_unembed"] = HarmoniTensor(name="t_Wunemb", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'], model['vocab']))
        self.weights_new["rms_norm3"] = HarmoniTensor(name="t_rmsnorm3", tag=HarmoniTensorType.WEIGHT, precision=model['dtype'], shape=(model['hdim'],))

    def update_weight_stride(self, layer, component, new_stride, head=None):
        """Updates the stride information for a specific weight."""
        if head is not None:
            self.weights_new[f"layer_{layer}"]["heads"][f"head_{head}"][component]["stride"] = new_stride
        else:
            self.weights_new[f"layer_{layer}"][component]["stride"] = new_stride
 
    def get_total_weight_size(self):
        """Calculates the total size of all model weights."""
        total_size_bytes = 0
        for layer, data in self.weights_new.items():
            if "layer" not in layer:
                total_size_bytes += data.size  # size_bytes
            else:
                for key, value in data.items():
                    if key in {"Wq", "Wk", "Wv"}:
                        for head, head_weights in value.items():
                            total_size_bytes += head_weights.size  # size_bytes
                    else:
                        total_size_bytes += value.size  # size_bytes

        total_size_KB = total_size_bytes / 1024
        total_size_MB = total_size_KB / 1024
        total_size_GB = total_size_MB / 1024
        return total_size_bytes, total_size_KB, total_size_MB, total_size_GB

    def print_weights(self):
            """Prints the weight metadata in a structured format."""
            print("------ WEIGHT INFO ------")
            print(f"Model Name: {self.model['name']}")
            print("Model Config:", self.model)
            for layer, data in self.weights_new.items():
                print(f"\n{layer}:")
                if "layer" not in layer:
                    print(f"  {layer}: {data}")
                else:
                    for key, value in data.items():
                        if key in {"Wq", "Wk", "Wv"}:
                            for head, wmeta in value.items():
                                print(f"  {key}_head{head}: {wmeta}")
                        else:
                            print(f"  {key}: {value}")
            print("------------------------")