# X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs

![X-MAS](./assets/xmas_overview.png)

## 🔥 News
- [2025/05/23] See our preprint paper in [ArXiv](https://arxiv.org/pdf/2505.16997).

## X-MAS-Bench

### 1. Specify your model configs in `./configs/X-MAS_Bench_config.json`:
```
"gpt-4o-mini-2024-07-18": {
        "model_list": [
            {"model_name": "gpt-4o-mini-2024-07-18", "model_url": "http://a.b.c.d:e/v1", "api_key": "xyz"}
        ],
        "max_workers_per_model": 10
    }
```

### 2. Inference on a dataset (the outputs will be saved under "./X-MAS-Bench/results/")
```
# bash scripts/infer_X-MAS_Bench.sh

python X-MAS-Bench/infer_qa.py --model_name <model_name> --model_config <config_path> --test_dataset_name <dataset_name>
```
If you want to infer with **revise, aggregation or evaluation**, please make sure you have replace the content of "./X-MAS-Bench/results/" with **source file** in [Google Drive](https://drive.google.com/file/d/1ijQCzujXdYZDV95vWJHUpvOQFWVgUbXF/view?usp=sharing) first. 
You can download the .zip file named results.zip to the "./X-MAS-Bench/results/" path and unzip it.

If you want to infer with **planning**, please make sure you have load the default source model **"llama-3.1-8b-instruct", "qwen-2.5-7b-instruct" and "qwen-2.5-14b-instruct"**.

### 3. Evaluate on a dataset (the outputs will be saved under "./X-MAS-Bench/results/")
```
# bash scripts/eval_X-MAS_Bench.sh

python X-MAS-Bench/eval_bench.py --eval_model_name <eval_model_name> --model_config <config_path> --dataset_name <dataset_name> --model_name <model_name> --function_name <function_name> --eval_mode bench-test
# We use llama-3.1-70b-instruct as <eval_model_name>
```

Note that we release the experimental results of the X-MAS-Bench in [Google Drive](https://drive.google.com/file/d/1oukYZLDOuc98i-ICkoZ6OYME9a7-AuH1/view?usp=drive_link).
You can download the .zip file named results.zip to the "./X-MAS-Bench/results/" path and unzip it.

## X-MAS-Design

### 1. Specify your model configs in `./configs/X-MAS_Design_config.json`:
```
"gpt-4o-mini-2024-07-18": {
        "model_list": [
            {"model_name": "gpt-4o-mini-2024-07-18", "model_url": "http://a.b.c.d:e/v1", "api_key": "xyz"}
        ],
        "max_workers_per_model": 10
    }
```

### 2. Inference on a dataset (the outputs will be saved under "./X-MAS-Design/results/")
```
# bash scripts/infer_X-MAS_Design.sh

# (Parallel)
python X-MAS-Design/inference_mas.py --model_api_config <model_api_config> --method_name <method_name> --test_dataset_name <test_dataset_name>


# Or (Sequential)
python X-MAS-Design/inference_mas.py --model_api_config <model_api_config> --method_name <method_name> --test_dataset_name <test_dataset_name> --sequential
```

You can change the default model list in the method_config files for the corresponding method, like model list in "./X-MAS-Design/methods/llm_debate/configs/config_main.yaml" for "./X-MAS-Design/methods/llm_debate/llm_debate_main.py"

If you want to run a method like llm_debate with your own config or settings, you can modify them accordingly and run the following code.

```
# bash scripts/infer_X-MAS_Design.sh

# (Parallel)
python X-MAS-Design/inference_mas.py --model_api_config <model_api_config> --method_name <method_name> --method_config_name <method_config_name> --test_dataset_name <test_dataset_name>

# Or (Sequential)
python X-MAS-Design/inference_mas.py --model_api_config <model_api_config> --method_name <method_name> --method_config_name <method_config_name> --test_dataset_name <test_dataset_name> --sequential
```

### 3. Evaluate on a dataset (the outputs will be saved under "./X-MAS-Design/results/")
```
# bash scripts/eval_X-MAS_Design.sh

python X-MAS-Design/eval_mas.py --eval_model_name llama-3.1-70b-instruct --model_api_config <model_api_config> --method_name <method_name> --method_config_name <method_config_name> --test_dataset_name <test_dataset_name> --eval_mode bench-test
```

## Citation

```
@article{ye2025x,
  title={X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs},
  author={Ye, Rui and Liu, Xiangrui and Wu, Qimin and Pang, Xianghe and Yin, Zhenfei and Bai, Lei and Chen, Siheng},
  journal={arXiv preprint arXiv:2505.16997},
  year={2025}
}
```
