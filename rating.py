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
from torch.nn import MSELoss
lossfn = MSELoss()

from dataloader import LoadDataset, LoadReviewData, process_template, preprocess_opt, opt_pure_tokenize,preprocess_opt_preference, LoadDataset_preference
from utils import now_time, get_random_template, ids2tokens, ids2tensors, custom_mapping
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast as autocast


accelerator = Accelerator()
rank = accelerator.process_index
device = accelerator.device

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', '-dr', type=str, default="./MoviesAndTV",
                    help='path for loading the pickle data')
parser.add_argument('--index', '-i', type=int, default=1,
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', '-b', type=int, default=96,
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
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument('--base_model', type=str, default="facebook/opt-2.7B")
parser.add_argument('--lambdat', type=float, default=0.5)
parser.add_argument('--tokenlen', type=int, default=32)
parser.add_argument('--dataset', type=str, default="amazon")
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

base_model = args.base_model

if base_model == "facebook/opt-2.7B":
    base_model_name = "facebook-opt-2.7B"
elif base_model == 'opt-125m':
    base_model_name = "facebook-opt-125M"
else:
    base_model_name = "facebook-opt-1.3B"
current_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
logging.basicConfig(level=logging.INFO, filename=f'../logs5/New_experiment_{args.dataset}_{current_time}_{args.prefix}.log', filemode='w')
print(f'../logs5/{current_time}_{args.prefix}.log')
lambda1 = args.lambdat
lambda2 = 1-lambda1
logging.info("Test template 2stage with new loss, weight: lambda1:{}, lambda2:{}".format(lambda1, lambda2))

pad_id = 1
from rating_module_opt import META2S

dataloader_num_workers = 0
ckpt_dir = '../checkpoint/'+args.dataset
prediction_path = os.path.join(ckpt_dir, "rating_UIRR_2stage_"+args.prefix+args.outf)
model_path = ckpt_dir+"/META_2stage_" + args.prefix + '_' + base_model_name

logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    logging.info('{:40} {}'.format(arg, getattr(args, arg)))
logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

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
processed_preference_path = f"{review_data.data_dir}/preference_{base_model_name}.pickle"

if not os.path.exists(processed_data_path):
    logging.info(now_time() +"Processing data...")
    preprocess_opt(review_data, base_model_name, tokenizer, bos, eos, pad_id)
    logging.info(now_time() +f"Processed data saved in {review_data.data_dir}/ex_reviews_{base_model_name}.pickle")

if not os.path.exists(processed_preference_path):
    logging.info(now_time() +"Processing preference data...")
    preprocess_opt_preference(review_data, base_model_name, tokenizer, bos, eos, pad_id)
    logging.info(now_time() +f"Processed data saved in {review_data.data_dir}/preference_{base_model_name}.pickle")


preference_dataset = LoadDataset_preference(processed_preference_path)
train_dataset = LoadDataset(data_dir, processed_data_path, args.index, "train_cold")
logging.info(now_time() +"Load train done")
valid_dataset = LoadDataset(data_dir, processed_data_path, args.index, "validation")
logging.info(now_time() +"Load valid done")

train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=dataloader_num_workers,
                # shuffle=True,
            )
valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                num_workers=dataloader_num_workers,
            )
preference_dataloader = DataLoader(
                preference_dataset,
                batch_size=int(args.batch_size/2),
                num_workers=dataloader_num_workers,
            )

ckpt_path = '../checkpoint/{}/NCFckpt.pth'.format(args.dataset)
# load model
model = META2S.from_pretrained(base_model, user_n, item_n, args.tokenlen, ckpt_path)

#model.resize_token_embeddings(len(tokenizer))
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

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


# load template
#template_path1 = f"{args.data_dir}/rating_prompt_UIRR_{base_model_name}_stage1.pickle"
#template_path2 = f"{args.data_dir}/rating_prompt_UIRR_{base_model_name}_stage2.pickle"
embedding = model.get_input_embeddings()







model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, valid_dataloader
)


def evaluate(dataloader):
    
    model.eval()
    total_loss = 0
    ttloss1 = 0
    ttloss2 = 0
    total_sample = 0
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
                
            if model.stage==2:
                outputs,rat = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids)
                ratingsum = torch.tensor([float(rating) for rating in batch[2]]).to(device).float().to(torch.int64)
                loss_2=lossfn(rat.squeeze(1), ratingsum)
                loss_1=outputs.loss
                loss = loss_1*lambda1+loss_2*lambda2
                ttloss1 += loss_1.item()
                ttloss2 += loss_2.item()
                total_loss += loss.item()
                total_sample += 1
                    
            else:
                
                outputs = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids)
                loss = outputs.loss
                total_loss += loss.item()
                total_sample += 1
        
    if model.stage==2:
        loss1=ttloss1/total_sample
        loss2=ttloss2/total_sample
        logging.info(now_time() + 'valid loss :loss1 {:4.4f} | loss2 {:4.4f}'.format(loss1, loss2))
        return total_loss/total_sample
    else:
        return total_loss/total_sample
            
    

logging.info(now_time() +"Stage {}:".format(model.stage))

model_path=model_path+'_stage1'


ckpt_path = None#'../checkpoint/yelprating_UIRR_2stage_38_facebook-opt-2.7B_stage1'

if ckpt_path:
    logging.info(now_time() +"Stage 1 has finished.")
    with open(ckpt_path, 'rb') as f:
        model = torch.load(f).to(device)
    unwrapped_model = accelerator.unwrap_model(model)
    model = accelerator.prepare(unwrapped_model)
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    best_val_loss = float('inf')
    model.train()
    endure_count = 0
    for epoch in range(args.epochs):
        logging.info(now_time() + 'epoch {}'.format(epoch))
        total_loss = 0
        total_sample = 0
        total_lossp = 0
        total_samplep = 0
        step = 0
        step_p=0
        

        for batch in train_dataloader:
            model.stage=1
            step+=1
            optimizer.zero_grad()
            with autocast():
                users = batch[0].to(device)
                items = batch[1].to(device)
                ratings = batch[2]
                ratings_id = torch.tensor([tokenizer.encode(rating) for rating in ratings]).to(device)
                input_ids = batch[3].squeeze(1).to(device)
                attention_mask = batch[4].squeeze(1).to(device)
                template_ids = batch[5].squeeze(1).to(device)
                outputs = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids)
                    

                loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            total_loss += loss.item()
            total_sample += 1

            if step % args.log_interval == 0:
                cur_t_loss = total_loss / total_sample
                logging.info(now_time() + 'loss {:4.4f} | {:5d} batches'.format(cur_t_loss, step))
                total_loss = 0
                total_sample = 0

        for batch in preference_dataloader:
            model.stage=4
            step_p+=1
            optimizer.zero_grad()
            with autocast():
                users = batch[0].to(device)
                items1 = batch[1].to(device)
                items2 = batch[1].to(device)
                input_ids = batch[3].squeeze(1).to(device)
                attention_mask = batch[4].squeeze(1).to(device)

                outputs = model.forward(users, items1, input_id=input_ids, mask=attention_mask, item2=items2)
                loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            total_lossp += loss.item()
            total_samplep+= 1
            if step_p % args.log_interval == 0:
                cur_t_loss = total_lossp / total_samplep
                logging.info(now_time() + 'Preference: loss {:4.4f} | {:5d} batches'.format(cur_t_loss, step_p))

        model.stage=1
        val_loss = evaluate(valid_dataloader)
        logging.info(now_time() + 'valid loss {:4.4f} on validation'.format( val_loss))
            
        # Save the model if the validation loss is the best we've seen so far.

        val_loss = torch.tensor([val_loss],device=device)
        val_loss = float(accelerator.reduce((val_loss))[0])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(model_path, 'wb') as f:
                    torch.save(model, f)
        else:
            endure_count += 1
            logging.info(now_time() + 'Endured {} time(s)'.format(endure_count))
            if endure_count == args.endure_times:
                logging.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                break



prompt = process_rating_template_opt(tokenizer, embedding, stage2_list)

model.stage=2

model_path=model_path+'_stage2'

best_val_loss = float('inf')



model.stage=2
model.init_net(device)
model.train()
endure_count = 0
logging.info(now_time() +"Stage {}:".format(model.stage))
for epoch in range(args.epochs):
    logging.info(now_time() + 'epoch {}'.format(epoch))
    l=0
    total_loss = 0
    ttloss1 = 0
    ttloss2 = 0
    total_sample = 0
    step = 0
    flag=0
    for batch in train_dataloader:
        step+=1
        optimizer.zero_grad()
        if flag==1:
            continue
        users = batch[0].to(device)
        items = batch[1].to(device)
        ratings = batch[2]
        ratings_id = torch.tensor([tokenizer.encode(rating) for rating in ratings]).to(device)
        input_ids = batch[3].squeeze(1).to(device)
        attention_mask = batch[4].squeeze(1).to(device)
        template_ids = batch[5].squeeze(1).to(device)
        if epoch>-1:
            outputs, rat = model.forward(users, items, ratings_id, input_ids, attention_mask, template_ids)
            ratingsum = torch.tensor([float(rating) for rating in batch[2]]).to(device).to(torch.int64)
            loss_2=lossfn(rat.squeeze(1).float(), ratingsum.float())
            loss_1=outputs.loss
            loss = loss_1*lambda1+loss_2*lambda2
            #if loss.item()<0.7:
            #    flag=1
                
            print('loss1 {:4.4f} | loss2 {:4.4f}'.format(loss_1.item(), loss_2.item()))
            
            
            ttloss1 += loss_1.item()
            ttloss2 += loss_2.item()
            total_sample += 1
            #assert loss.item()<15
            
        else:
            outputs, rat = model.forward(users, items, ratings_id, input_ids, attention_mask, prompt)
            loss_1=outputs.loss
            loss = loss_1
            print(loss_1.item())
            total_sample += 1
        accelerator.backward(loss)
        
        optimizer.step()
        total_loss += loss.item()
        

        if step % args.log_interval == 0:
            cur_t_loss = total_loss / total_sample
            loss1= ttloss1/total_sample
            loss2= ttloss2/total_sample
            logging.info(now_time() + 'loss1 {:4.4f} | loss2 {:4.4f} | total_loss {:4.4f} | {:5d} batches'.format(loss1, loss2, cur_t_loss, step))
            total_loss = 0
            ttloss1 = 0
            ttloss2 = 0
            total_sample = 0
    if epoch>1:
        l=1
    #lambda2 = 1
    val_loss = evaluate(valid_dataloader)
    logging.info(now_time() + 'valid loss {:4.4f} on validation'.format( val_loss))
        
    # Save the model if the validation los

    val_loss = torch.tensor([val_loss],device=device)
    val_loss = float(accelerator.reduce((val_loss))[0])
    
    
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
        
    else:
        endure_count += 1
        logging.info(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            logging.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break   