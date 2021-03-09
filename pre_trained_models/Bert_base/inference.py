from pytorch_pretrained_bert import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import torch
import numpy as np


def load_model(use_gpu: bool=True):
    model = BertModel.from_pretrained('bert-base-uncased')
    if use_gpu:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


def generate_vecs(models, document, use_gpu: bool=True):
    if use_gpu:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    model, tokenizer = models

    embedding = []
    for sent in document:
        tokenized_sent = tokenizer.tokenize(sent)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
        segments_ids = [0] * len(tokenized_sent)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        embedding.append(np.mean(encoded_layers[-1][0].cpu().numpy(), axis=0))

    return embedding
