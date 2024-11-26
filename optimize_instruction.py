
from pathlib import Path
import argparse
import json

from datasets import DatasetDict, load_dataset

from infer import evolver, evaluator, optimizer

def get_prompt(prompt_dir: str) -> str:
    # 評価用プロンプトの読み込み
    with open(Path(prompt_dir, "evaluation_prompt.txt"), "r") as f:
        evaluation_prompt = f.read()
    
    # 初期の進化用プロンプトの読み込み
    with open(Path(prompt_dir, "initial_evolution_prompt.txt"), "r") as f:
        initial_evolution_prompt = f.read()

    # プロンプト最適化用のプロンプトの読み込み
    with open(Path(prompt_dir, "optimization_prompt.txt"), "r") as f:
        optimization_prompt = f.read()
    
    return evaluation_prompt, initial_evolution_prompt, optimization_prompt

def get_dataset(dataset_name: str, split: str, column_name: str) -> DatasetDict:
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(lambda x: {"base_instruction": x[column_name]})
    return dataset

def run_optimize_prompt(args, dataset: DatasetDict):
    # 最適化処理の対象となるサブセットを取得
    sub_dataset = dataset.shuffle(seed=42).select(range(args.evaluation_size))

    # プロンプトの読み込み
    evaluation_prompt, initial_evolution_prompt, optimization_prompt = get_prompt(args.prompt_dir)

    # 初期の進化用プロンプトでの評価
    batch_sub_dataset = sub_dataset.map(lambda x: {"evolved_instruction": evolver(x["base_instruction"], args.model_name, initial_evolution_prompt)})
    batch_sub_dataset = batch_sub_dataset.map(lambda x: {"evaluation": evaluator(x["base_instruction"], x["evolved_instruction"], args.model_name, evaluation_prompt)})
    best_score = sum(batch_sub_dataset["evaluation"])
    with open(Path(args.output_dir, f"log.jsonl"), "a") as f:
        f.write(json.dumps({"step": 0, "score": best_score, "prompt": initial_evolution_prompt}) + "\n")

    evolution_prompt = initial_evolution_prompt

    # 進化処理の実行
    for step in range(args.max_steps):
        result = []
        for i, _ in enumerate(range(args.repeat_optimization)):
            # 進化用プロンプトの最適化
            optimized_prompt = optimizer(evolution_prompt, args.model_name, optimization_prompt)
            
            # 進化用プロンプトでの評価
            batch_sub_dataset = sub_dataset.map(lambda x: {"evolved_instruction": evolver(x["base_instruction"], args.model_name, optimized_prompt)})
            batch_sub_dataset = batch_sub_dataset.map(lambda x: {"evaluation": evaluator(x["base_instruction"], x["evolved_instruction"], args.model_name, evaluation_prompt)})
            batch_sub_dataset.to_json(Path(args.output_dir, "steps", f"step_{step}_repeat_{i}.jsonl"))
            score = sum(batch_sub_dataset["evaluation"])
            result.append({"step": step, "score": score, "prompt": optimized_prompt})
        
        # 結果を保存
        with open(Path(args.output_dir, f"log.jsonl"), "a") as f:
            for r in result:
                f.write(json.dumps(r) + "\n")

        # resultの中で最もスコアが高いものを選択
        best_result = max(result, key=lambda x: x["score"])
        print(f"Step {step} Done. Best Score: {best_result['score']}")
        if best_result["score"] <= best_score:
            break
        else:
            evolution_prompt = best_result["prompt"]
            best_score = best_result["score"]
    
    with open(Path(args.output_dir, "best_prompt.txt"), "w") as f:
        f.write(evolution_prompt)
    print(f"Evolution Done. Best Score: {best_score}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="train")
    parser.add_argument("--column_name", type=str, required=True, default="instruction")
    parser.add_argument("--evaluation_size", type=int, required=True, default=100)
    parser.add_argument("--max_steps", type=int, required=True, default=3)
    parser.add_argument("--repeat_optimization", type=int, required=True, default=3)
    parser.add_argument("--batch_size", type=int, required=True, default=4)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name, args.split, args.column_name)

    run_optimize_prompt(args, dataset)

if __name__ == "__main__":
    main()