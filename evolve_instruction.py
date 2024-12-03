
from pathlib import Path
import argparse
import json

from datasets import DatasetDict, load_dataset

from infer import evolver

def get_prompt(prompt_path: str) -> str:
    with open(Path(prompt_path), "r") as f:
        evolution_prompt = f.read()
    
    return evolution_prompt

def get_dataset(dataset_name: str, split: str, column_name: str) -> DatasetDict:
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.map(lambda x: {"base_instruction": x[column_name]})
    return dataset

def run_evolve_instruction(args, dataset: DatasetDict):
    evolution_prompt = get_prompt(args.prompt_dir)
    dataset = dataset.map(lambda x: {"evolved_instruction": evolver(x["base_instruction"], args.model_name, evolution_prompt)}, num_proc=args.batch_size)
    dataset.to_json(Path(args.output_dir, f"evolved_instruction.jsonl"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, default="huggingface")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="train")
    parser.add_argument("--column_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True, default=8)

    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name, args.split, args.column_name)

    run_evolve_instruction(args, dataset)

if __name__ == "__main__":
    main()