# extract reddit data into sentences and save it to csv
import pandas as pd
import json

path_comments = "../data/reddit_comments.json"
path_labels =  "../data/reddit_train-balanced.csv"

def read_comments(path_c = path_comments, path_l = path_labels):
    json_data = open(path_comments).read()
    data = json.loads(json_data)
    
    res = []
    
    with open(path_labels) as f:
        
        for line in f:
            _, ids, labels = line.rstrip("\n").split("|")
            ids, labels = list(ids.split(" ")), list(labels.split(" "))
            comments = list(zip(ids, labels))
            
            for comment, label in (comments):
                res.append([comment, data[comment]["text"], label ])
    
    df = pd.DataFrame(res, columns=['c_id', 'text', 'label'])
    df.to_csv("../data/reddit_dataset.csv")
            
        
        


    
read_comments(path_c = path_comments, path_l = path_labels)