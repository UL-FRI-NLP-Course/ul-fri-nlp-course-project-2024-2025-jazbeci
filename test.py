# from transformers import AutoConfig
# config = AutoConfig.from_pretrained("cjvt/GaMS-9B-Instruct")
# print(config)

"""
gams config is:
Gemma2Config {
  "architectures": [
    "Gemma2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": 50.0,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "final_logit_softcapping": 30.0,
  "head_dim": 256,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "gemma2",
  "num_attention_heads": 16,
  "num_hidden_layers": 42,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "sliding_window_size": 4096,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "vocab_size": 256000
}

"""

from transformers import pipeline

model_id = "cjvt/GaMS-9B-Instruct"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",  # Automatically distribute across GPUs and CPU
)

# Example of response generation
message = [{"role": "user", "content": "Kateri je najpomembnejši dogodek v slovenski zgodovini?"}]
response = pline(message, max_new_tokens=512)
try:
    print("Model's response:", response[0]["generated_text"][-1]["content"])
except KeyError:
    print("Model's response:", response[0]["generated_text"])
except Exception as e:
    # Print the full response
    print("Model's response:", response)
    print("Error:", e)

# Example of conversation chain
new_message = response[0]["generated_text"]
new_message.append({"role": "user", "content": "Lahko bolj podrobno opišeš ta dogodek?"})
response = pline(new_message, max_new_tokens=1024)
try:
    print("Model's response:", response[0]["generated_text"][-1]["content"])
except KeyError:
    print("Model's response:", response[0]["generated_text"])
except Exception as e:
    # Print the full response
    print("Model's response:", response)
    print("Error:", e)