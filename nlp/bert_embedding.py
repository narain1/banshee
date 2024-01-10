from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModel.from_pretrained("")
encoded_input = tokenizer("", return_tensors='pt')
model_output = model(**encoded_input)

token_embeddings = model_output['last_hidden_state']

mean_embedding = torch.mean(token_embeddings, dim=1)

normalized_embeddings = F.normalize(mean_embedding)


