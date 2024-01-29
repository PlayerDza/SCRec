from datasets import load_dataset
from accelerate import Accelerator
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers import TrainingArguments, AdamW, Trainer
import evaluate
from torch.utils.data import DataLoader
from torch import nn
import torch
import argparse
import time
import logging
import os
import math
import random
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


from dataloader import LoadDataset,LoadDataset_topn, LoadReviewData, process_template, preprocess, opt_pure_tokenize
from utils import now_time, get_random_template, ids2tokens, ids2tensors, str2int, evaluate_hr, evaluate_ndcg
from torch.nn import MSELoss
lossfn = MSELoss()
accelerator = Accelerator()
rank = accelerator.process_index
device = accelerator.device

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', '-dr', type=str, default="/data/huliya/Amazon/MoviesAndTV",
                    help='path for loading the pickle data')
parser.add_argument('--index', '-i', type=int, default=1,
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', '-b', type=int, default=64,
                    help='batch size')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='gpu number')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/',
                    help='directory to save the final model')
parser.add_argument('--prefix', type=str, default='',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=3,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument('--base_model', type=str, default="gpt2")
parser.add_argument('--lambdat', type=float, default=0.5)
parser.add_argument('--tokenlen', type=int, default=32)
parser.add_argument('--dataset', type=str, default="amazon")
args = parser.parse_args()

current_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
logging.basicConfig(level=logging.INFO, filename=f'../logs5/Test_New_experiment_topn_{args.dataset}_{current_time}_{args.prefix}.log', filemode='w')
logging.info("Test template w/o rating exp")
base_model = args.base_model
if base_model == "facebook/opt-2.7B":
    base_model_name = "facebook-opt-2.7B"
elif base_model == 'opt-125m':
    base_model_name = "facebook-opt-125M"
else:
    base_model_name = "facebook-opt-1.3B"
from rating_module_opt import META2S
dataloader_num_workers = 0
ckpt_dir = '../checkpoint/'+args.dataset
prediction_path = os.path.join(ckpt_dir, "rating_UIRR_2stage_"+args.prefix+args.outf)
model_path = ckpt_dir+"/META_2stage_topn_" + args.prefix + '_' + base_model_name

lambda1 = args.lambdat
lambda2 = 1-lambda1

# load tokenizer
pad = '<pad>'
bos = '<s>'
eos = '</s>'
tokenizer = AutoTokenizer.from_pretrained(base_model, bos_token=bos, eos_token=eos, pad_token=pad)
# tokenizer.pad_token = pad


# prepare dataset
logging.info(now_time() +"Loading dataset...")
data_dir = '../data/'+args.dataset
review_data = LoadReviewData(data_dir)
user_n = len(review_data.user_dict)
item_n = len(review_data.item_dict)
logging.info(now_time() +"Load review done")
processed_topdata_path = f"{data_dir}/topn.pickle"

user2item_gt = pickle.load(open(f"{data_dir}/topn_gt.pickle",'rb'))
test_dataset = LoadDataset_topn(processed_topdata_path)
logging.info(now_time() +"Load test done")

test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=dataloader_num_workers,
            )

# load model

# load template
#template_path = f"{args.data_dir}/rating_prompt_UIRR_{base_model_name}_stage2.pickle"
#prompt = pickle.load(open(template_path, 'rb'))

def process_rating_template_opt(tokenizer, embedding, rating_template):
    data = []
    
    embedding = embedding.cpu()
    
    for tmp in rating_template:
        input_ids = opt_pure_tokenize(tokenizer, tmp)
        input_ids = torch.tensor(input_ids)
        input_embs = embedding(input_ids)
        data.append(input_embs)

    embedding.to(device) 
    return data


def evaluate(dataloader):
    model.eval()
    total_loss = 0
    total_sample = 0
    ttloss1 = 0
    ttloss2 = 0
    with torch.no_grad():
        for batch in dataloader:
            users = batch[0].to(device)
            items = batch[1].to(device)
            ratings = batch[2]
            ratings_id = torch.tensor([tokenizer.encode(rating) for rating in ratings]).to(device)
            input_ids = batch[3].squeeze(1).to(device)
            attention_mask = batch[4].squeeze(1).to(device)
            template_ids = batch[5].squeeze(1).to(device)
                #outputs = model.forward(users, items, ratings_id, prompt)
            #print(ratings_id)
            outputs,rat = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids)
            ratingsum = torch.tensor([float(rating) for rating in batch[2]]).to(device).to(torch.int64)
            loss_2=lossfn(rat.squeeze(1), ratingsum)
            loss_1=outputs.loss
            loss = loss_1*lambda1+loss_2*lambda2
            ttloss1 += loss_1.item()
            ttloss2 += loss_2.item()
            total_loss += loss.item()
            total_sample += 1


    loss1=ttloss1/total_sample
    loss2=ttloss2/total_sample
    logging.info(now_time() + 'valid loss :loss1 {:4.4f} | loss2 {:4.4f}'.format(loss1, loss2))    
    return total_loss/total_sample

def inference():
    model.eval()
    predict_list = []
    
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            cnt+=1
            
            users = batch[0].to(device).to(torch.int64)
            items = batch[1].to(device).to(torch.int64)
            user_list=users.cpu().tolist()
            item_list=items.cpu().tolist()
            
            ratings = batch[2]
            #input_ids = batch[3].squeeze(1).to(device)
            #attention_mask = batch[4].squeeze(1).to(device)
            ratings_id = torch.tensor([tokenizer.encode(rating)[:1] for rating in ratings]).to(device)
            #print(cnt)
            #template_ids = batch[5].squeeze(1).to(device)
            # if(cnt!=362):
            #     continue
                #print(user_list)
                #print(item_list)
            
            outputs = model.forward(users, items, ratings_id, inference=True)
            
            
            #print(cnt)
            rating_list=outputs.cpu().tolist()
            #print(rating_list)
            #logging.info(user_list,item_list,rating_list)
            assert len(user_list)==len(item_list) and len(user_list)==len(rating_list)
            
            for i,u in enumerate(user_list):
                predict_list.append({
                    'user':u,
                    'item':item_list[i],
                    'rating':rating_list[i][0]
                })
            
            
            

    return predict_list


ckpt_path = '../checkpoint/{}/NCFckpt.pth'.format(args.dataset)
model = META2S.from_pretrained(base_model, user_n, item_n, args.tokenlen, ckpt_path)

#model.resize_token_embeddings(len(tokenizer))
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr)
model.stage=3
embedding = model.get_input_embeddings()
#prompt = process_rating_template_opt(tokenizer, embedding, test_prompt)

model, optimizer, test_dataloader = accelerator.prepare(
        model, optimizer, test_dataloader
    )


model_pathi=model_path+'_stage1'+'_stage2'
    # Load the best saved model.
with open(model_pathi, 'rb') as f:
    model = torch.load(f).to(device)
unwrapped_model = accelerator.unwrap_model(model)
model = accelerator.prepare(unwrapped_model)

model.to(device)
predicts = inference()
logging.info('=' * 89)
predicts.sort(key=lambda x: (x['user'], x['rating']),reverse=True)

pickle_file_path = 'topn_pred.pickle'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(predicts, file)
user2rank_list={}

for i in predicts:
    if str(i['user']) not in user2rank_list.keys():
        user2rank_list[str(i['user'])]=[str(i['item'])]
    else:
        user2rank_list[str(i['user'])].append(str(i['item']))




if rank == 0:
    top_ns = [1]

    for i in range(1, (10 // 5) + 1):
            top_ns.append(i * 5)
    for top_n in top_ns:
        hr = evaluate_hr(user2item_gt, user2rank_list, top_n)
        logging.info(now_time() + 'HR@{} {:7.4f}'.format(top_n, hr))
        ndcg = evaluate_ndcg(user2item_gt, user2rank_list, top_n)
        logging.info(now_time() + 'NDCG@{} {:7.4f}'.format(top_n, ndcg))




