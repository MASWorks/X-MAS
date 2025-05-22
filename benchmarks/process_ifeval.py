import json
from copy import deepcopy

original_file_path = "/mnt/petrelfs/yerui/mac/HeteroMAS/evaluation/ifeval/data/input_data.jsonl"
save_file_path = "./test_pool/IFEval.json"

with open(original_file_path, "r") as f:
    data = [json.loads(line) for line in f]

save_data = []
for sample in data:
    tmp = {}
    tmp["query"] = sample["prompt"]
    del sample["prompt"]
    tmp.update(sample)
    tmp["tag"] = ["instruction following", "IFEval", "general"]
    tmp["source"] = "IFEval"
    save_data.append(tmp)

with open(save_file_path, "w") as f:
    json.dump(save_data, f, indent=4)