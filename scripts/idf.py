import os,pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm.auto import tqdm
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
model_type_or_dir="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
collection_picked_file="/raid/datasets/shared/MSMARCO/collection.pickle"
if os.path.exists(collection_picked_file):
    with open(collection_picked_file, 'rb') as f:
        corpus = pickle.load(f)
# listCorp=list(corpus.values())
listCorp=[]
corpusLen=len(corpus)
for key in tqdm(range(corpusLen),desc="reading data"):
    listCorp.append(corpus[str(key)])
batch=100000
numBatch=corpusLen//batch+1
tokenCop=[]
for idx in tqdm(range(0,numBatch),desc="tokening"):
    if idx*batch>corpusLen:
        break
    tokenCop+=list(tokenizer(listCorp[idx*batch:(idx+1)*batch], return_tensors="np")['input_ids'])
# tokenCop=tokenizer(listCorp, return_tensors="np")['input_ids']
InverseVocab={key[1]:key[0] for ind,key in enumerate(tokenizer.vocab.items())}
numToken=len(tokenizer.vocab.keys())
idf={ind:0 for ind in range(numToken)}
idfToken={key:0 for key in tokenizer.vocab}

for tokenSent in tqdm(tokenCop,desc="processing data"):
    SetTokenSent=set(tokenSent)
    for token in SetTokenSent:
        idf[token]+=1
        idfToken[InverseVocab[token]]+=1
minFreq=10
for key in idf:
    if idf[key]<minFreq:
        idf[key]=minFreq
with open('output/idf-tokenid.pkl', 'wb') as f:
    pickle.dump(idf, f)
with open('output/idf-token.pkl', 'wb') as f:
    pickle.dump(idfToken, f)

        