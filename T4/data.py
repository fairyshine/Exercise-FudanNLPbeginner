import json

def getDataset(file_path):
    Dataset=[]
    with open(file_path,'r') as f:
        for line in f:
            LS=[]
            data=json.loads(line)
            LS.append(data['text'].lower().split())
            LS.append(data['NER'].split())
            Dataset.append(LS)
    return Dataset