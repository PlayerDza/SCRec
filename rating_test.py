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


from dataloader import LoadDataset, LoadReviewData, process_template, preprocess, opt_pure_tokenize
from utils import now_time, get_random_template, ids2tokens, ids2tensors, str2int
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
logging.basicConfig(level=logging.INFO, filename=f'../logs5/Test_New_experiment_{args.dataset}_{current_time}_{args.prefix}.log', filemode='w')
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
model_path = ckpt_dir+"/META_2stage_" + args.prefix + '_' + base_model_name

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
processed_data_path = f"{review_data.data_dir}/ex_reviews_{base_model_name}.pickle"

if not os.path.exists(processed_data_path):
    logging.info(now_time() +"Processing data...")
    preprocess(review_data, base_model_name, tokenizer, bos, eos)
    logging.info(now_time() +f"Processed data saved in {review_data.data_dir}/reviews_{base_model_name}.pickle")
test_dataset = LoadDataset(data_dir, processed_data_path, args.index, "test")
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

test_prompt = [
    " is chosen by ", ".How will ", "rate ", "?(1 being lowest and 5 being highest)"
]


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
    predicts = []
    labels = []
    cnt = 0
    with torch.no_grad():
        for batch in test_dataloader:
            cnt+=1
            users = batch[0].to(device).to(torch.int64)
            items = batch[1].to(device).to(torch.int64)
            ratings = batch[2]
            input_ids = batch[3].squeeze(1).to(device)
            attention_mask = batch[4].squeeze(1).to(device)
            ratings_id = torch.tensor([tokenizer.encode(rating)[:1] for rating in ratings]).to(device)
            template_ids = batch[5].squeeze(1).to(device)
            outputs = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids, inference=True)
            ratings = torch.tensor([float(rating) for rating in batch[2]]).to(device).to(torch.int64)
            labels.extend(ratings.cpu().numpy())
            predicts.extend(outputs.cpu().numpy())

    return labels, predicts
#model_path='../checkpoint/yelprating_UIRR_2stage_38_facebook-opt-2.7B_stage1_stage2'

ckpt_path = '../checkpoint/{}/NCFckpt.pth'.format(args.dataset)
model = META2S.from_pretrained(base_model, user_n, item_n, args.tokenlen, ckpt_path)
#model.resize_token_embeddings(len(tokenizer))
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)
model.stage=2
embedding = model.get_input_embeddings()
prompt = process_rating_template_opt(tokenizer, embedding, test_prompt)

model, optimizer, test_dataloader = accelerator.prepare(
        model, optimizer, test_dataloader
    )


model_pathi=model_path+'_stage1'+'_stage2'
    # Load the best saved model.
with open(model_pathi, 'rb') as f:
    model = torch.load(f).to(device)
unwrapped_model = accelerator.unwrap_model(model)
model = accelerator.prepare(unwrapped_model)
#logging.info('This is epoch {}:'.format(epoch))
    # Run on test data.
test_loss = evaluate(test_dataloader)
test_loss = torch.tensor([test_loss], device=device)
test_loss = float(accelerator.reduce((test_loss))[0])
if rank == 0:
    logging.info('=' * 89)
    logging.info(now_time() + 'loss {:4.4f} on test | End of training'.format(test_loss))
    logging.info(now_time() + 'Generating text')
labels, predicts = inference()

if rank == 0:
    with open(prediction_path, 'w', encoding='utf-8') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]) + " " + str(predicts[i]) + "\n")
    logging.info(now_time() + 'Generated text saved to ({})'.format(prediction_path))

    rmse = mean_squared_error(labels, predicts, squared=False)
    mae = mean_absolute_error(labels, predicts)
    logging.info(now_time() + 'RMSE {:4.4f}'.format(rmse))
    logging.info(now_time() + 'MAE {:4.4f}'.format(mae))




