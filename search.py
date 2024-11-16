import argparse
import hashlib
import numpy as np
import os

from pathlib import Path
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import namedtuple


MODEL = 'intfloat/multilingual-e5-base'


def iterate_markdown_files(vault_path):
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                yield file_path


def load_text(path):
    with open(path, 'r') as f:
        return f.read()


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(input_texts):
    # Each input text should start with "query: " or "passage: ", even for non-English texts.
    # For tasks other than retrieval, you can simply use the "query: " prefix.

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().numpy()


Row = namedtuple('Row', ('pos', 'hash'))


def scan_files(vault_path, index_path):
    filepaths = list(iterate_markdown_files(vault_path))

    are_files_updated = False
    embeddings = []
    if Path(index_path).exists():
        index = np.load(index_path)
        paths = index['path']
        hashes = index['hash']
        embeddings = index['embedding']

        index = {path: Row(pos, hash) for pos, (path, hash) in enumerate(zip(paths, hashes))}
    else:
        index = {}
        are_files_updated = True

    embeds_to_add = []

    for path in tqdm(filepaths):
        text = 'passage: ' + load_text(path)
        hash = hashlib.sha256(text.encode()).hexdigest()
        row = index.get(path)
        if not row or row.hash != hash:
            embedding = get_embeddings([text])
            are_files_updated = True
            if not row:
                pos = len(embeddings) + len(embeds_to_add)
                embeds_to_add.append(embedding)
                index[path] = Row(pos, hash)
            else:
                p = row.pos
                embedding = np.ravel(embedding)
                embeddings[p, :] = embedding
            
    if not are_files_updated:
        return paths, embeddings

    embeddings = [embeddings] + embeds_to_add if len(embeddings) else embeds_to_add
    embeddings = np.vstack(embeddings)

    n = len(embeddings)
    filepaths = np.zeros(n, dtype='<U512')
    hashes = np.zeros(n, dtype='<U64')
    
    for path, row in index.items():
        filepaths[row.pos] = path
        hashes[row.pos] = row.hash

    np.savez(index_path, embedding=embeddings, path=filepaths, hash=hashes)
    return filepaths, embeddings


def score_files(filenames, embeddings, query, ntop):
    query = 'query: ' + query
    qemb = get_embeddings([query])
    sim = np.dot(embeddings, qemb.T)
    sim = np.ravel(sim)
    pos = np.argsort(sim)[-ntop:][::-1]
    for p in pos:
        score = sim[p]
        print('Document:', filenames[p], f'({score})') 



# Run the function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=3, help='Number of documents to search')
    parser.add_argument('--index-path', type=str, default='index.npz')
    parser.add_argument('-v', '--vault-path', required=True, help='Path to Obsidian vault directory')
    parser.add_argument('query', help='Query to search for in notes')
    args = parser.parse_args()
    paths, embeddings = scan_files(args.vault_path, args.index_path)
    score_files(paths, embeddings, args.query, args.n)
