# 微调后推理
# CUDA_VISIBLE_DEVICES= python inference_codellama.py

import os
import sys

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

import jsonlines

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            prompt,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=512,
            # stream_output=False,
            **kwargs,
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            # repetition_penalty=2.0,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                # do_sample=True,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output

    start = 6395
    # end = 0
    with jsonlines.open(f'../CodeCritic/dataset/raw_data_test.jsonl', 'r') as reader:
        for id, line in enumerate(reader, start=5826):
            if id < start:
                continue

            # if id > end:
            #     break

            role = (
                "You are an expert code analyzer and will be provided with a piece of code for an algorithm question. "
                "Please analyze the code according to the following evaluation criteria to evaluate the code quality. ")

            criteria = (
                "Criteria: (1)Is there any compilation error in the code? (2)Is the code functionally correct? (3)Is there an "
                "algorithm that is more efficient than the one used by the code? (4)Is the code too long or not "
                "concise enough? (5)Can the code judgment structure be simplified? (6)What is the cyclomatic "
                "complexity of the code? Is it too high? (7)What is the cognitive complexity of the code? Is it too "
                "high? (8)Are there any bad smells in the code? If so, please point them out.")

            system_prompt = role + criteria

            user_message = f"### Question:\n{line['question']}\n\n### Code:\n{line['code']}\n\n### Feedback:"

            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

            # print("\n\nResponse:\n", evaluate(full_prompt))
            print(f"id: {id}")

            data = {
                "id": line["id"],
                "contestId": line["contestId"],
                "index": line["index"],
                "chat": evaluate(full_prompt)
            }

            with jsonlines.open("../CodeCritic/dataset/test_result.jsonl", mode='a') as writer:
                writer.write(data)


if __name__ == "__main__":
    fire.Fire(main)
