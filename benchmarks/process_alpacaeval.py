import json

original_path = "../../datasets/alpaca_eval.json"
save_path = "./test_pool/AlpacaEval.json"

with open(original_path, "r") as f:
    data = json.load(f)

save_data = []
for sample in data:
    save_data.append(
        {
            "query": sample["instruction"],
            "tag": ["AlpacaEval", sample["dataset"]],
            "source": "AlpacaEval"
        }
    )

with open(save_path, "w") as f:
    json.dump(save_data, f, indent=4)