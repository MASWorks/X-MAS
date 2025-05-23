# X-MAS: Diversity for The Win: Towards Building Multi-Agent Systems with Heterogeneous LLMs


## Get Started

### X-MAS-Bench

1. Specify your model configs in `./configs/X-MAS_Bench_config.json`:
```
"deepdeek-r1-distill-qwen-14b": [
    ["deepdeek-r1-distill-qwen-14b", "http://a.b.c.d:e/v1", "xyz"]
]
```

2. To inference on a dataset/several datasets(The output_path will be "./X-MAS-Bench/results/")
```
# Or Step 2 (Parallel): Inference on several datasets
bash script/infer_X-MAS_Bench.sh
```

3. To evaluate on a dataset/several datasets(The output_path will be "./X-MAS-Bench/results/")
```
# Step 1: evaluate on several datasets
bash script/eval_X-MAS_Bench.sh
```

4. To get the results of our X-MAS-Bench in the paper
```
The link to the experimental results of the X-MAS-Bench is here https://drive.google.com/file/d/1oukYZLDOuc98i-ICkoZ6OYME9a7-AuH1/view?usp=drive_link. 
Please download the .zip file named results.zip to the "./X-MAS-Bench/results/" path and unzip it
```


### X-MAS-Design

1. Specify your model configs in `./configs/X-MAS_Design_config.json`:
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
python X-MAS-Design/inference_X-MAS.py --method_name <method_name> --debug
```

3. To inference on a dataset/several datasets(The output_path will be "./results/")
```
# Step 1: build the test dataset
python datasets/build_test_dataset.py --dataset_name <dataset_name>

# Step 2 (Sequential): Inference on the whole dataset
python X-MAS-Design/inference_X-MAS.py --method_name <method_name> --test_dataset_name <dataset_name> --sequential

# Or Step 2 (Parallel): Inference on the whole dataset
python X-MAS-Design/inference_X-MAS.py --method_name <method_name> --test_dataset_name <dataset_name>

# Or Step 2 (Parallel): Inference on several datasets
bash script/infer_X-MAS_Design.sh
```

4. To evaluate on a dataset/several datasets(The output_path will be "./results/")
```
# Step 1: evaluate on several datasets
bash script/eval_X-MAS_Design.sh
```