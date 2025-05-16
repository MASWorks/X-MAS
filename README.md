# X-MAS: Diversity for The Win: Towards Building Multi-Agent Systems with Heterogeneous LLMs


## Get Started

1. Specify your model configs in `./model_api_configs/model_api_config.json`:
```
"gpt-4o-mini-2024-07-18": {
        "model_list": [
            {"model_name": "gpt-4o-mini-2024-07-18", "model_url": "http://a.b.c.d:e/v1", "api_key": "xyz"}
        ],
        "max_workers_per_model": 10
    }
```

2. To see if the codebase is executable (e.g., vanilla, cot, agentverse)
```
python inference.py --method_name <method_name> --debug
```

3. To inference on a dataset
```
# Step 1: build the test dataset
python datasets/build_test_dataset.py --dataset_name <dataset_name>

# Step 2 (Sequential): Inference on the whole dataset
python inference.py --method_name <method_name> --test_dataset_name <dataset_name> --sequential

# Or Step 2 (Parallel): Inference on the whole dataset
python inference.py --method_name <method_name> --test_dataset_name <dataset_name>
```