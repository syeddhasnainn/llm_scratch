import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# each row represents a token and each of them is represented by a 3D vector or embeddings
print(inputs.shape)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)

print(inputs)
for i, x_i in enumerate(inputs):
    #for x_0 = [0.43, 0.15, 0.89].[0.55, 0.87, 0.66]
    attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
print(attn_scores_2)