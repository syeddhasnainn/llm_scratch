import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
# each row represents a token and each of them is represented by a 3D vector or embeddings
print(inputs.shape)

query = inputs[1]

#initialized an empty tensor with inputs.shape[0] = 6
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    #for x_0 = [0.43, 0.15, 0.89].[0.55, 0.87, 0.66]
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

#normalizing the attention score to get attention weights

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print('weights:', attn_weights_2_tmp)
print('sum:', attn_weights_2_tmp.sum())

#pytorch softmax

print(attn_scores_2.shape)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print('attention weights',attn_weights_2)

#computing context vector for input 2 = z2
query = inputs[1] 
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

#calculating all attention scores
#matrix multiplication
attn_scores = inputs @ inputs.T

attn_weights = torch.softmax(attn_scores, dim=1)
print('all attention weights:',attn_weights)

#all context vectors

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)