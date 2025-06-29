# 🦙 LLaMA Finetuning: Domain-Specific Language Model Adaptation

This project demonstrates fine-tuning Meta's [LLaMA](https://ai.meta.com/llama/) language model on custom domain data using the Hugging Face `transformers` and `peft` libraries. The focus is on leveraging **parameter-efficient fine-tuning (LoRA)** to adapt large language models for specific downstream tasks with minimal compute overhead.

## 🚀 Project Highlights

- Dataset used: https://huggingface.co/datasets/whoispanashe/medquad-guanco-llama2
- ✅ Fine-tunes LLaMA models using **LoRA (Low-Rank Adaptation)**
- ✅ Utilizes Hugging Face's `transformers`, `datasets`, `peft`, and `bitsandbytes`
- ✅ Supports INT4 quantization for memory-efficient training
- ✅ Suitable for domain-specific NLP tasks with minimal resource requirements
- ✅ Structured for easy deployment and experimentation

## 🛠️ Tech Stack

- Python 3.10+
- [Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT](https://huggingface.co/docs/peft/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- Google Colab or local GPU (NVIDIA A100 preferred for full fine-tuning)

## 📂 Project Structure

```
Llama_Finetuning.ipynb  # Complete notebook for data prep, training, and evaluation
README.md               # Project documentation
```

## 🔍 Model Details

- **Base Model**: LLaMA (7B or 13B)
- **Finetuning Method**: LoRA (rank=8, alpha=16)
- **Precision**: 4-bit quantized weights using `bnb_config`
- **Tokenizer**: LLaMA Tokenizer from HuggingFace
- **Training Data**: Custom domain-specific text dataset

## 📈 Training Configuration

| Parameter        | Value          |
|------------------|----------------|
| LoRA Rank        | 8              |
| LoRA Alpha       | 16             |
| LoRA Dropout     | 0.05           |
| Batch Size       | 4              |
| Epochs           | 3              |
| Max Length       | 2048 tokens    |
| Gradient Accum.  | 4              |
| Optimizer        | AdamW          |

## 📦 Installation

Make sure to install the following libraries before running the notebook:

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## ▶️ Usage

Run the notebook `Llama_Finetuning.ipynb` step-by-step:

1. Load and prepare your dataset.
2. Load base LLaMA model and tokenizer.
3. Apply LoRA for parameter-efficient adaptation.
4. Train the model using `transformers` Trainer.
5. Evaluate and save the fine-tuned model.

## 📌 Results

The fine-tuned LLaMA model demonstrates improved performance on domain-specific prompts. LoRA drastically reduces GPU memory consumption and training time while preserving accuracy.

## 📚 References

- Meta AI's [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)
- Hugging Face [PEFT](https://huggingface.co/docs/peft/index)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---
