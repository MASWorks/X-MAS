import json

original_path = "../../datasets/arena-hard-v0.1.jsonl"
save_path = "./test_pool/Arena-Hard-v0.1.json"

with open(original_path, "r") as f:
    data = [json.loads(line) for line in f]

save_data = []
for sample in data:
    save_data.append(
        {
            "query": sample["turns"][0]["content"],
            "tag": ["Arena-Hard-v0.1", sample["category"], sample["cluster"]],
            "source": "Arena-Hard-v0.1"
        }
    )

with open(save_path, "w") as f:
    json.dump(save_data, f, indent=4)