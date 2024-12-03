import re

import backoff

from openai import OpenAI

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def evolver(instruction: str, model_name: str, prompt: str):
    openai_api_key = OPENAI_API_KEY
    openai_api_base = OPENAI_API_BASE
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    temperature = 0.5
    max_tokens = 2048

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは与えられたinstructionをより複雑なものに書き換えるアシスタントです。"},
            {"role": "user", "content": prompt.replace("INSTRUCTION", instruction)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 最終的に書き換えられたinstructionを取得
    rewritten_instruction = re.findall(r"<finally_rewritten_instruction>(.*?)</finally_rewritten_instruction>", response.choices[0].message.content, re.DOTALL)[-1]

    return rewritten_instruction

@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def evaluator(base_instruction: str, evolved_instruction: str, model_name: str, prompt: str):
    openai_api_key = OPENAI_API_KEY
    openai_api_base = OPENAI_API_BASE
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    temperature = 0.0
    max_tokens = 1024

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは与えられた2つのinstructionを比較して、より複雑なinstructionに書き換えられているか評価するアシスタントです。"},
            {"role": "user", "content": prompt.replace("BASE_INSTRUCTION", base_instruction).replace("EVOLVED_INSTRUCTION", evolved_instruction)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 評価結果を取得
    evaluation = int(response.choices[0].message.content.split(": ")[1])

    # 1/0以外の値が返ってきた場合はエラー
    if evaluation not in [1, 0]:
        evaluator(base_instruction, evolved_instruction, model_name, prompt)

    return evaluation

def optimizer(base_prompt: str, model_name: str, prompt: str) -> str:
    openai_api_key = OPENAI_API_KEY
    openai_api_base = OPENAI_API_BASE
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    temperature = 0.9
    max_tokens = 2048

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは与えられたinstructionをより複雑なものに書き換えるアシスタントです。"},
            {"role": "user", "content": prompt.replace("PROMPT", base_prompt)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 最適化されたプロンプトを取得
    optimized_prompt = re.search(r"<prompt>(.*)</prompt>", response.choices[0].message.content, re.DOTALL).group(1).strip()

    # 必要な要素が含まれていない場合は再実行
    if "INSTRUCTION" not in optimized_prompt:
        optimizer(base_prompt, model_name, prompt)
    
    if "<finally_rewritten_instruction>" not in optimized_prompt and "</finally_rewritten_instruction>" not in optimized_prompt:
        optimizer(base_prompt, model_name, prompt)
    
    return optimized_prompt