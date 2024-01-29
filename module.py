from transformers import OPTForCausalLM
import torch.nn as nn
import torch
import copy
import random
from torch.nn import Module


class NCF(Module):

    def __init__(self,userNum,itemNum,dim,layers=[128,64,32,8]):
        super(NCF, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.fc_layers = torch.nn.ModuleList()
        self.finalLayer = torch.nn.Linear(layers[-1],1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.fc_layers.append(nn.Linear(From,To))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)
        x = embd
        for l in self.fc_layers:
            x = l(x)
            x = nn.ReLU()(x)

        return uembd,iembd

class MLP(torch.nn.Module):
    def __init__(self, emb_dim1, emb_dim2):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim1, 100),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(100, emb_dim2))

    def forward(self, input):
        output = self.decoder(input)
        return output


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim1, emb_dim2):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim1, 100), torch.nn.ReLU(),
                                           torch.nn.Linear(100, emb_dim1 * emb_dim2))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input):
        output = self.decoder(input)
        return output

class Mapping_u(torch.nn.Module):
    def __init__(self, nuser, nitem, ckpt_path):
        super().__init__()
        self.ncf(nuser,nitem, ckpt_path)
        self.map = MetaNet(64, 2560)

    def ncf(self, nuser, nitem, ckpt_path):
        self.ncfmodel = NCF(nuser,nitem,64,layers=[128,64,32,16,8])
        self.ncfmodel.load_state_dict(torch.load(ckpt_path))
        for param in self.ncfmodel.parameters():
            param.requires_grad = False

    def forward(self, user, item):
        u_ebd,i_ebd = self.ncfmodel(user, item).unsqueeze(1)
        
        mapping = self.map(u_ebd).view(-1, 64, 2560)
        
        u_src = torch.bmm(u_ebd, mapping).squeeze(1)
        
        
        return u_src

class Mapping_i(torch.nn.Module):
    def __init__(self, nuser, nitem, ckpt_path):
        super().__init__()
        self.ncf(nuser,nitem, ckpt_path)
        self.map = MetaNet(64, 2560)

    def ncf(self, nuser, nitem, ckpt_path):
        self.ncfmodel = NCF(nuser,nitem,64,layers=[128,64,32,16,8])
        self.ncfmodel.load_state_dict(torch.load(ckpt_path))
        for param in self.ncfmodel.parameters():
            param.requires_grad = False

    def forward(self, user, item):
        u_ebd,i_ebd = self.ncfmodel(user, item).unsqueeze(1)
        
        mapping = self.map(i_ebd).view(-1, 64, 2560)
        
        i_src = torch.bmm(i_ebd, mapping).squeeze(1)
        
        
        return i_src

class meta2S:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, token, ckpt, freezeLM=True, **kwargs):
        
        model = super().from_pretrained(pretrained_model_name_or_path,   **kwargs)
        
        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        model.init_prompt2(nuser, nitem)
        model.init_softtoken(token)
        model.init_softtoken2(token)
        model.init_softtokenp(token)
        model.stage=1
        
        model.ckpt_path=ckpt
        model.init_map1(nuser, nitem)
        model.init_map2(nuser, nitem)
        
        return model

    def init_map_u(self, nuser, nitem):
        emsize = self.model.decoder.embed_tokens.weight.size(1)
        device = self.model.device
        self.mapu = Mapping_u(nuser, nitem, self.ckpt_path).to(device)
        
    
    def init_map_i(self, nuser, nitem):
        emsize = self.model.decoder.embed_tokens.weight.size(1)
        device = self.model.device
        self.mapi = Mapping_i(nuser, nitem, self.ckpt_path).to(device)
        

    

    def init_softtoken(self, n_tokens = 12):
        self.learned_param = nn.parameter.Parameter(self.model.decoder.embed_tokens.weight[:n_tokens].clone().detach())
        
        initrange = 0.1
        
        self.learned_param.data.uniform_(-initrange, initrange)
        
    def init_softtoken2(self, n_tokens = 12):
        self.learned_param2 = nn.parameter.Parameter(self.model.decoder.embed_tokens.weight[:n_tokens].clone().detach())
        
        initrange = 0.1
        
        self.learned_param2.data.uniform_(-initrange, initrange)
        
    
    def init_softtokenp(self, n_tokens = 12):
        self.learned_paramp = nn.parameter.Parameter(self.model.decoder.embed_tokens.weight[:n_tokens].clone().detach())

        initrange = 0.1
        
        self.learned_paramp.data.uniform_(-initrange, initrange)
        

    def init_prompt(self, nuser, nitem):
        self.src_len = 2
        emsize = self.model.decoder.embed_tokens.weight.size(1)  # 768
        self.user_embeddings1 = nn.Embedding(nuser, emsize)
        self.item_embeddings1 = nn.Embedding(nitem, emsize)
        

        initrange = 0.1
        self.user_embeddings1.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings1.weight.data.uniform_(-initrange, initrange)

    def init_net(self, device):
        self.mlp = MLP(5,1).to(device)
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def init_net2(self, device):
        self.mlp2 = MLP(2,1).to(device)
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_prompt2(self, nuser, nitem):
        emsize = self.model.decoder.embed_tokens.weight.size(1)
        self.user_embeddings2 = nn.Embedding(nuser, emsize)
        self.item_embeddings2 = nn.Embedding(nitem, emsize)
        initrange = 0.1
        self.user_embeddings2.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings2.weight.data.uniform_(-initrange, initrange)


    def forward(self, user, item, rating=None, input_id=None, mask=None, template_id=None, inference=False, item2=None, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)



        soft_token1 = self.learned_param.repeat(batch_size, 1, 1).to(device)
        
        if self.stage==1:
            
            u_src1 = self.user_embeddings1(user)  # (batch_size, emsize)
            i_src1 = self.item_embeddings1(item)  # (batch_size, emsize)
            t_src = self.model.decoder.embed_tokens(template_id)
            w_src = self.model.decoder.embed_tokens(input_id)
            rating_src = self.model.decoder.embed_tokens(rating)
            
            src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), t_src], 1)
            src_len=src.size(1)
            
            src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), t_src, w_src], 1)
            if inference:
                 # (batch_size, total_len, emsize)
                return super().forward(inputs_embeds=src)
            else:
                pad_left = torch.ones((batch_size, src_len), dtype=torch.int64).to(device)
                pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

                # prediction for training
                pred_left = torch.full((batch_size, src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
                pred_right = torch.where(mask == 1, input_id, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
                prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
                
                return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

        elif self.stage==2:
            rating_src = self.model.decoder.embed_tokens(rating)
            self.user_embeddings1.weight.requires_grad = False
            self.item_embeddings1.weight.requires_grad = False
            self.learned_param.requires_grad = False
            
            ####
            
            soft_token2 = self.learned_param2.repeat(batch_size, 1, 1).to(device)
            soft_token1 = self.learned_param.repeat(batch_size, 1, 1).to(device)
            ####
            u_src1 = self.user_embeddings1(user)  # (batch_size, emsize)
            i_src1 = self.item_embeddings1(item)  # (batch_size, emsize)
            u_src2 = self.mapu(user)  # (batch_size, emsize)
            i_src2 = self.mapi(item)  # (batch_size, emsize)
            src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), soft_token2,  u_src2.unsqueeze(1), i_src2.unsqueeze(1), rating_src], 1)
            none_rating_len = src.size(1)-rating_src.size(1)
            
            if inference:
                
                src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), soft_token2,  u_src2.unsqueeze(1), i_src2.unsqueeze(1), rating_src], 1)
                outputs = super().forward(inputs_embeds=src)
                
                sliced_x = outputs.logits[:, none_rating_len:, :]
                
                last_token = outputs.logits[:, -1, :]
                
                logits1 = last_token[:,134]
                logits2 = last_token[:,176]
                logits3 = last_token[:,246]
                logits4 = last_token[:,306]
                logits5 = last_token[:,245]
                logits_num = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1), logits3.unsqueeze(1), logits4.unsqueeze(1), logits5.unsqueeze(1)], 1)
                
                return self.mlp(logits_num)
            else:
                
                
                pad_input = torch.ones((batch_size, src.size(1)), dtype=torch.int64).to(device)

                
                pred_left = torch.full((batch_size, none_rating_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
                prediction = torch.cat([pred_left, rating], 1).to(device)  # (batch_size, total_len)
                
                outputs = super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)
                sliced_x = outputs.logits[:, none_rating_len:-1, :]
                
                weights = torch.linspace(1, 0, outputs.logits.size(1)-none_rating_len-1)
                weights = weights.view(1, outputs.logits.size(1)-none_rating_len-1, 1).expand_as(sliced_x).to(sliced_x.device)
                
                weighted_x = sliced_x * weights
                pooled_last_token = weighted_x.mean(dim=1)
                
                last_token = outputs.logits[:, -2, :]
                
                word_prob = torch.softmax(pooled_last_token, dim=-1)
                predict_token = torch.argmax(word_prob, dim=1, keepdim=True)
                
                logits1 = last_token[:,134]
                logits2 = last_token[:,176]
                logits3 = last_token[:,246]
                logits4 = last_token[:,306]
                logits5 = last_token[:,245]
                logits_num = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1), logits3.unsqueeze(1), logits4.unsqueeze(1), logits5.unsqueeze(1)], 1)
                
                return outputs, self.mlp(logits_num)
        elif self.stage==3:
            rating_src = self.model.decoder.embed_tokens(rating)
            
            self.user_embeddings1.weight.requires_grad = False
            self.item_embeddings1.weight.requires_grad = False
            self.learned_param.requires_grad = False
            
            ####
            
            soft_token2 = self.learned_param2.repeat(batch_size, 1, 1).to(device)
            soft_token1 = self.learned_param.repeat(batch_size, 1, 1).to(device)
            ####
            u_src1 = self.user_embeddings1(user)  # (batch_size, emsize)
            i_src1 = self.item_embeddings1(item)  # (batch_size, emsize)
            u_src2 = self.mapu(user)  # (batch_size, emsize)
            i_src2 = self.mapi(item)  # (batch_size, emsize)
            src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), soft_token2,  u_src2.unsqueeze(1), i_src2.unsqueeze(1), rating_src], 1)
            none_rating_len = src.size(1)-rating_src.size(1)
            
            if inference:
                
                src = torch.cat([soft_token1, u_src1.unsqueeze(1), i_src1.unsqueeze(1), soft_token2,  u_src2.unsqueeze(1), i_src2.unsqueeze(1), rating_src], 1)
                outputs = super().forward(inputs_embeds=src)
                
                sliced_x = outputs.logits[:, none_rating_len:, :]
                
                last_token = outputs.logits[:, -1, :]
                
                logits1 = last_token[:,134]
                logits2 = last_token[:,288]
                
                logits_num = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], 1)
                #print("*")
                return self.mlp(logits_num)
            else:
                
                pad_input = torch.ones((batch_size, src.size(1)), dtype=torch.int64).to(device)

                
                pred_left = torch.full((batch_size, none_rating_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
                prediction = torch.cat([pred_left, rating], 1).to(device)  # (batch_size, total_len)
                
                outputs = super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)
                sliced_x = outputs.logits[:, none_rating_len:-1, :]
                
                weights = torch.linspace(1, 0, outputs.logits.size(1)-none_rating_len-1)
                weights = weights.view(1, outputs.logits.size(1)-none_rating_len-1, 1).expand_as(sliced_x).to(sliced_x.device)
                
                weighted_x = sliced_x * weights
                pooled_last_token = weighted_x.mean(dim=1)
                
                last_token = outputs.logits[:, -2, :]
                
                
                
                logits1 = last_token[:,134] #1
                logits2 = last_token[:,288] #0
                
                logits_num = torch.cat([logits1.unsqueeze(1), logits2.unsqueeze(1)], 1)
                
                return outputs, self.mlp(logits_num)
        elif self.stage==4:
            u_src1 = self.user_embeddings1(user)  # (batch_size, emsize)
            i_src11 = self.item_embeddings1(item)  # (batch_size, emsize)
            i_src12 = self.item_embeddings1(item2)  # (batch_size, emsize)
            soft_tokenp = self.learned_paramp.repeat(batch_size, 1, 1).to(device)
            w_src = self.model.decoder.embed_tokens(input_id)
            ui_src1 = self.map1(user, item)
            ui_src2 = self.map1(user, item2)
            src = torch.cat([soft_tokenp, u_src1.unsqueeze(1), i_src11.unsqueeze(1), i_src12.unsqueeze(1)], 1)
            src_len=src.size(1)
            
            src = torch.cat([soft_tokenp, u_src1.unsqueeze(1), i_src11.unsqueeze(1), i_src12.unsqueeze(1), w_src], 1)
            if inference:
                 # (batch_size, total_len, emsize)
                return super().forward(inputs_embeds=src)
            else:
                pad_left = torch.ones((batch_size, src_len), dtype=torch.int64).to(device)
                pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

                # prediction for training
                pred_left = torch.full((batch_size, src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
                pred_right = torch.where(mask == 1, input_id, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
                prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
                
                return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)



class META2S(meta2S, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)


