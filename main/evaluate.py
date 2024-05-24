import torch
from gpt import GPTModel, generate_text_simple
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,     
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    #squeezing because the model expect batch of sequence, shape [1,4]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding('gpt2')

token_ids = generate_text_simple(model=model, 
                                 idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=10,
                                 context_size=GPT_CONFIG_124M['context_length'])

print('output text:\n', token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100], [40,    1107, 588]])
targets = torch.tensor([[3626, 6100, 345  ], [588,  428,  11311]])

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas,dim=-1, keepdim=True)
print('Token IDs:\n',token_ids)

print(f'target batch 1', token_ids_to_text(targets[0], tokenizer))
print('output batch 1', token_ids_to_text(token_ids[0].flatten(), tokenizer))

text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print('text 1', target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print('text 2', target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print('logits shape', logits.shape)
print('targets shape', targets.shape)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

print('flattened logits', logits_flat.shape)
print('flattened tarets', targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

perplexity = torch.exp(loss)
print(perplexity)