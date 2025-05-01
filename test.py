from transformers import AutoConfig
config = AutoConfig.from_pretrained("cjvt/GaMS-9B-Instruct")
print(config)