from torch.utils import data
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, OPTForCausalLM

import pickle
import os
import random
import torch



amazon_template = ["What will {user} say about selecting {item}, which cost {price}$ and belongs to {cat}?",
                   "What are {suer}'s thoughts on selecting the {item}, which is a {cat} product, for {price} dollars?",
                   "How does {user} feel about the choice of {item}, under the {cat} category, and costing {price}$?",
                   "What's {user}'s stance on acquiring {item}, which belongs to the {cat} category and costs {price}$?",
                   "How would {user} comment on choosing {item}, priced at {price} dollars and part of the {cat} category?",
                   "{user}'s opinion is sought on the selection of {item}, a {cat} product, with a price point of {price} dollars. What is it?"
                   ]

trip_template = ["What will {user} from {location} say about selecting the item?",
                 "What would {user} from {location} have to say regarding the choice of the item?",
                 "How does {user} in {location} feel about the selection of the item?",
                 "What would {user} from {location} have to say regarding the choice of the item?",
                 "How is {user} in {location} likely to describe their feelings about choosing the item?"]

yelp_template = ["What will {user} say about choosing to live in {item}, which locates at {add}, {city}, {state}, {post}?",
                 "How would {user} describe their choice to live in {item} at {add}, {city}, {state}, {post}?",
                 "What opinion does {user} have about settling in {item} at {add}, {city}, {state}, {post}?",
                 "How does {user} feel about the choice of living in {item} situated at {add}, {city}, {state}, {post}?",
                 "How is {user} likely to comment on choosing to inhabit {item} at {add}, {city}, {state}, {post}?"]




def preprocess_opt(review_data, base_model, tokenizer, bos, eos, pad_id, max_seq_len=25, word_len=20):
    pad = pad_id
    # max_len = 0
    all_review_data = []
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    reviews = pickle.load(open(f"{review_data.data_dir}/ex_reviews.pickle", 'rb'))
    for review in reviews:
        text = review['review']
        tokens = opt_pure_tokenize(tokenizer, text)
        text = tokenizer.decode(tokens[:word_len])
        user = review['user']
        item = review['item']
        rating = review['rating']
        output=tokenizer(f"{bos} {text} {eos}", return_tensors='pt', truncation=True, max_length=max_seq_len)
        input_ids = output['input_ids'][:,1:]
        attention_mask = output['attention_mask']
        feature = review['feature']
        #print(input_ids.size())
        if 'amazon' in review_data.data_dir:
            temp = random.choice(amazon_template).replace("{user}", review['username']).replace("{item}", review['itemtitle']).replace("{price}", str(review['price'])).replace("{cat}", review['categories'])
        elif 'yelp' in review_data.data_dir:
            temp = random.choice(yelp_template).replace("{user}", review['username']).replace("{item}", review['itemname']).replace("{add}", str(review['address'])).replace("{city}", review['city']).replace("{state}", str(review['state'])).replace("{post}", review['postal_code'])
        else:
            temp = random.choice(trip_template).replace("{user}", review['username']).replace("{item}", review['location'])
        
        temp_output = tokenizer.encode(temp, return_tensors='pt', max_length=30, padding="max_length", truncation=True)
        
        temp_id = temp_output[:,1:]
        # if len(input_ids[0]) > max_len:
        #     max_len = len(input_ids[0])
        if len(input_ids[0]) < max_seq_len:
            input_ids = torch.cat((input_ids, torch.tensor([pad]).repeat(1, max_seq_len - len(input_ids[0]))), 1)
            attention_mask = torch.cat((attention_mask, torch.zeros(1, max_seq_len - len(attention_mask[0]), dtype=torch.int32)), 1)
        all_review_data.append({'user': review_data.user_dict.entity2idx[user],
                    'item': review_data.item_dict.entity2idx[item],
                    'template_ids': temp_id,
                    'rating': rating,
                    'text': text,
                    'feature': feature,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask})
    pickle.dump(all_review_data, open(f"{review_data.data_dir}/ex_reviews_{base_model}.pickle", 'wb'))

def preprocess_opt2(review_data, base_model, tokenizer, bos, eos, pad_id, max_seq_len=25, word_len=20):
    pad = pad_id
    # max_len = 0
    all_review_data = []
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    reviews = pickle.load(open(f"{review_data.data_dir}/topn_reviews.pickle", 'rb'))
    for review in reviews:
        text = review['review']
        tokens = opt_pure_tokenize(tokenizer, text)
        text = tokenizer.decode(tokens[:word_len])
        user = review['user']
        item = review['item']
        rating = review['rating']
        output=tokenizer(f"{bos} {text} {eos}", return_tensors='pt', truncation=True, max_length=max_seq_len)
        input_ids = output['input_ids'][:,1:]
        attention_mask = output['attention_mask']
        feature = review['feature']
        #print(input_ids.size())
        if 'amazon' in review_data.data_dir:
            temp = random.choice(amazon_template).replace("{user}", review['username']).replace("{item}", review['itemtitle']).replace("{price}", str(review['price'])).replace("{cat}", review['categories'])
        elif 'yelp' in review_data.data_dir:
            temp = random.choice(yelp_template).replace("{user}", review['username']).replace("{item}", review['itemname']).replace("{add}", str(review['address'])).replace("{city}", review['city']).replace("{state}", str(review['state'])).replace("{post}", review['postal_code'])
        else:
            temp = random.choice(trip_template).replace("{user}", review['username']).replace("{item}", review['location'])
        
        temp_output = tokenizer.encode(temp, return_tensors='pt', max_length=30, padding="max_length", truncation=True)
        
        temp_id = temp_output[:,1:]
        # if len(input_ids[0]) > max_len:
        #     max_len = len(input_ids[0])
        if len(input_ids[0]) < max_seq_len:
            input_ids = torch.cat((input_ids, torch.tensor([pad]).repeat(1, max_seq_len - len(input_ids[0]))), 1)
            attention_mask = torch.cat((attention_mask, torch.zeros(1, max_seq_len - len(attention_mask[0]), dtype=torch.int32)), 1)
        all_review_data.append({'user': review_data.user_dict.entity2idx[user],
                    'item': review_data.item_dict.entity2idx[item],
                    'template_ids': temp_id,
                    'rating': rating,
                    'text': text,
                    'feature': feature,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask})
    pickle.dump(all_review_data, open(f"{review_data.data_dir}/topn_reviews_{base_model}.pickle", 'wb'))


def preprocess_opt_preference(review_data, base_model, tokenizer, bos, eos, pad_id, max_seq_len=50, word_len=20):
    pad = pad_id
    # max_len = 0
    all_review_data = []
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    reviews = pickle.load(open(f"{review_data.data_dir}/preference.pickle", 'rb'))
    for review in reviews:
        text = review['review']
        tokens = opt_pure_tokenize(tokenizer, text)
        text = tokenizer.decode(tokens[:word_len])
        user = review['user']
        item1 = review['item1']
        item2 = review['item2']
        #rating = review['rating']
        output=tokenizer(f"{bos} {text} {eos}", return_tensors='pt', truncation=True, max_length=max_seq_len)
        input_ids = output['input_ids'][:,1:]
        attention_mask = output['attention_mask']
        #feature = review['feature']
        #print(input_ids.size())
        # if 'amazon' in review_data.data_dir:
        #     temp = random.choice(amazon_template).replace("{user}", review['username']).replace("{item}", review['itemtitle']).replace("{price}", str(review['price'])).replace("{cat}", review['categories'])
        # elif 'yelp' in review_data.data_dir:
        #     temp = random.choice(yelp_template).replace("{user}", review['username']).replace("{item}", review['itemname']).replace("{add}", str(review['address'])).replace("{city}", review['city']).replace("{state}", str(review['state'])).replace("{post}", review['postal_code'])
        # else:
        #     temp = random.choice(trip_template).replace("{user}", review['username']).replace("{item}", review['location'])
        
        # temp_output = tokenizer.encode(temp, return_tensors='pt', max_length=30, padding="max_length", truncation=True)
        
        #temp_id = temp_output[:,1:]
        # if len(input_ids[0]) > max_len:
        #     max_len = len(input_ids[0])
        if len(input_ids[0]) < max_seq_len:
            input_ids = torch.cat((input_ids, torch.tensor([pad]).repeat(1, max_seq_len - len(input_ids[0]))), 1)
            attention_mask = torch.cat((attention_mask, torch.zeros(1, max_seq_len - len(attention_mask[0]), dtype=torch.int32)), 1)
        all_review_data.append({'user': review_data.user_dict.entity2idx[user],
                    'item1': review_data.item_dict.entity2idx[item1],
                    'item2': review_data.item_dict.entity2idx[item2],
                    #'template_ids': temp_id,
                    #'rating': rating,
                    'text': text,
                    #'feature': feature,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask})
    pickle.dump(all_review_data, open(f"{review_data.data_dir}/preference_{base_model}.pickle", 'wb'))

class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)
    
class LoadReviewData():
    def __init__(self, data_dir, max_seq_len=20):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.seq_len = max_seq_len
        self.data_dir = data_dir
        self.feature_set = set()

        reviews = pickle.load(open(f"{data_dir}/ex_reviews.pickle", 'rb'))
        # update max/min rating and dict
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            self.feature_set.add(review['feature'])
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating




        
class LoadDataset(data.Dataset):
    def __init__(self, data_dir, data_path, index, mode='train'):
        all_review_data = pickle.load(open(data_path, 'rb'))

        self.data = []
        self.feature = []

        with open(f"{data_dir}/index/{index}/{mode}.index", 'r') as f:
            index_list = [int(x) for x in f.readline().split(' ')]
        for idx in index_list:
            tmp = all_review_data[idx]
            self.feature.append(tmp["feature"])
            self.data.append(tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["user"], self.data[index]["item"], str(self.data[index]["rating"]), self.data[index]["input_ids"], self.data[index]["attention_mask"], self.data[index]["template_ids"]

    def get_input_ids(self):
        return [x["input_ids"] for x in self.data]
    
    def get_ratings(self):
        return [x["rating"] for x in self.data]
    
    def get_feature(self):
        return self.feature

class LoadDataset_topn(data.Dataset):
    def __init__(self, data_path):
        self.data = pickle.load(open(data_path, 'rb'))
        for i in self.data:
            i['rating']=0
            
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["user"], self.data[index]["item"], str(self.data[index]["rating"])



class LoadDataset_preference(data.Dataset):
    def __init__(self, data_path):
        self.data = pickle.load(open(data_path, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["user"], self.data[index]["item1"], str(self.data[index]["item2"]), self.data[index]["input_ids"], self.data[index]["attention_mask"]



def opt_pure_tokenize(tokenizer, text):
    input_ids = tokenizer.encode(text)
    return input_ids[1:]

class LoadDataset2(data.Dataset):
    def __init__(self, data_dir, data_path, index, gen_list, tokenizer, bos, eos, pad_id, mode='train', max_seq_len=25 , word_len=20):
        all_review_data = pickle.load(open(data_path, 'rb'))

        self.data = []
        self.feature = []

        with open(f"{data_dir}/index/{index}/{mode}.index", 'r') as f:
            index_list = [int(x) for x in f.readline().split(' ')]
        for idx in index_list:
            tmp = all_review_data[idx]
            self.feature.append(tmp["feature"])
            self.data.append(tmp)
        pad = pad_id
        for i in range(len(gen_list)):
            #print(len(self.data))
            #print(len(gen_list))
            #print(self.data[i])
            assert len(gen_list)==len(self.data)
            text = gen_list[i]
            tokens = opt_pure_tokenize(tokenizer, text)
            text = tokenizer.decode(tokens[:word_len])
            output=tokenizer(f"{bos} {text} {eos}", return_tensors='pt', truncation=True, max_length=max_seq_len)
            input_ids = output['input_ids'][:,1:]
            attention_mask = output['attention_mask']
            if len(input_ids[0]) < max_seq_len:
                input_ids = torch.cat((input_ids, torch.tensor([pad]).repeat(1, max_seq_len - len(input_ids[0]))), 1)
                attention_mask = torch.cat((attention_mask, torch.zeros(1, max_seq_len - len(attention_mask[0]), dtype=torch.int32)), 1)
            self.data[i]['text'] = text
            self.data[i]['input_ids'] = input_ids
            self.data[i]['attention_mask'] = attention_mask
            #print(self.data[i])
            



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["user"], self.data[index]["item"], str(self.data[index]["rating"]), self.data[index]["input_ids"], self.data[index]["attention_mask"]

    def get_input_ids(self):
        return [x["input_ids"] for x in self.data]
    
    def get_ratings(self):
        return [x["rating"] for x in self.data]
    
    def get_feature(self):
        return self.feature