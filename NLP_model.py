### apply language model

from sentence_transformers import SentenceTransformer, util, losses, models, InputExample, SentencesDataset, evaluation
from torch.utils.data import DataLoader,TensorDataset
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import utl

# Pretrain中用于处理Tokenized的类
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

# 模型输出中使用的人工检查功能，一共4个
def _Get_selected_Scenario_(sim,F_data,S_data):
    # 按需求，使用模型计算的sim获取相似度最高的10个scenario
    # 根据计算的相似度排序，保存所有scenario中前10个
    order_results = {}
    for k in range(len(F_data)):
        order_results[F_data[k]] = []
        tem = sim[F_data[k]].copy()
        tem_S = S_data.copy()
        while len(tem)!=0 and len(order_results[F_data[k]])<=10:
            target = max(tem)
            imax = tem.index(target)
            order_results[F_data[k]].append(tem_S[imax])
            tem.pop(imax)
            tem_S.pop(imax)
    return order_results

def _Get_searched_Scenario_(F_data,ref_data):
    # 按需求，使用参考数据获取scenario
    # 在给定ref_data中找到feature，并将数据中对应scenario返回保存
    order_results = {}
    for k in range(len(F_data)):
        order_results[F_data[k]] = []
        human_link = ref_data.copy()[:,0].tolist()
        human_link_s = ref_data.copy()[:,1].tolist() # 获取参考表中记录
        while F_data[k] in human_link: # 逐一匹配，找到人工对应表的S文本
            ind = human_link.index(F_data[k])
            order_results[F_data[k]].append(human_link_s[ind])
            human_link.pop(ind)
            human_link_s.pop(ind)
    return order_results

def _Get_matched_Scenario_(F_data,ref_data):
    # 按需求，统计与feature中含有相同文字的scenario
    # 给定feature，将与feature中有相同文字的scenario返回保存
    # SS_data为所有的scenario保存的列表
    SS_data = list(set(ref_data.copy()[:,1].tolist())) # 获取参考表中记录
    order_results = {}
    for k in range(len(F_data)):
        order_results[F_data[k]] = []
        for sitem in SS_data:
            for fitem in F_data[k]:
                if fitem in sitem:
                    order_results[F_data[k]].append(sitem)
    return order_results

def _Get_BM25_Scenario(k1,k2,b,F_data,SS_data):
    # 按需求，利用BM2.5模型筛选scenario，作为NLP模型辅助结果，参考https://zhuanlan.zhihu.com/p/79202151
    # 给定feature，通过BM2.5将scenario中与feature相近的词语返回保存
    # 每个f的每个词都可以对s统计一个词频，因此一个f含有多个词频，最终取词频最高的s作为输出
    # 由于S作为文档太短，会出现dft为0的情况，因此对W做修正
    # SS_data为所有的scenario保存列表，k1 k2取1.2~2，b取值0.75，
    
    def get_tf(f,SS_data):
        TSS_data = ''
        for s in SS_data:
            TSS_data+=s
        count = {}
        for s in TSS_data:
            count[f] = count.get(f,0)+1
        return count[f]

    def get_df(f,SS_data):
        count = 0
        for s in SS_data:
            if f in s:
                count+=1
        return count
    
    Lave = np.mean([len(s) for s in SS_data]) # 文档平均长度
    N = len(SS_data) # 所有文档个数
    # 统计词频，每一个f包含多个词，因此对应多个词频
    results={}
    for f in F_data:
        results[f]={}
        for s in SS_data:#所有文档
            Ld = len(s) # 当前文档长度
            results[f][s]=0
            tem1 = [get_tf(itf,SS_data) for itf in f] # itf在SS中的词频
            tem2 = [get_tf(itf,s) for itf in f] # itf在s中的词频
            tem3 = [get_df(itf,SS_data) for itf in f] # 包含itf的文档数
            for tftd,tftq,dft in zip(tem1,tem2,tem3):
                S = ((k2+1)*tftq)/(k2+tftq)
                W = np.log(N/(1+dft))*((k1+1)*tftd/(k1*((1-b)+b*(Ld/Lave))+tftd))
                results[f][s] += S*W            

    # 返回每个f在上述词频统计中最高的s
    order = {}
    for f in F_data:
        num = [results[f][s] for s in SS_data]
        order[f]=[SS_data[num.index(max(num))]]
    
    return order


class NLP_sim:
    
    def __init__(self,config):
        self.config=config
        self._load_model_()
    
    def _load_model_(self,):
        # Load one of the bert model as initial model
        if self.config.initial:
            ## 选择一个已有语言模型
            self.model = SentenceTransformer(self.config.pretrainFile + self.config.model_name)
        else:
            ## 选择一个已有语言模型
            self.model = SentenceTransformer(self.config.finetuneFile +  self.config.model_name)
    
    def _GetEmbedding_(self,words):
        # 计算文本的embedding
        return self.model.encode(words)
    
    def _GetSim_cos_(self,em1,em2):
        # 计算两个embedding的相似度
        return util.cos_sim(em1,em2)[0]
    
    def _finetune_eval_(self,model_link):
        # Get the fine-tune curl
        eval = pd.read_csv(model_link)
        return eval
    
    def _table_check_(self,F_data,S_data,sim):
        # 通过参考表进行人工检测输出结果检验
        human_link = self.config.ref_data.copy()[:,0].tolist()
        human_link_s = self.config.ref_data.copy()[:,1].tolist() # 获取参考表中记录
        while F_data in human_link: # 逐一匹配，找到人工对应表的S文本
            ind = human_link.index(F_data)
            stext = human_link_s[ind]
            if type: #选择是否需要进行相似度的归一化计算与相似扩大计算，如果是扩大则直接将人工检验匹配的结果乘上1.5并通过最大值上限1截断，反之直接取1
                sim[F_data][S_data.index(stext)] = np.min([sim[F_data][S_data.index(stext)]*self.config.max_expand,1.00]) 
            else:
                sim[F_data][S_data.index(stext)] = 1.00
            human_link.pop(ind)
            human_link_s.pop(ind)


    def Get_SimMatrix(self,type,F_data,S_data,savefilename):
        # Input two sets of data to obtain sim_matrix
        embeddings_F = self._GetEmbedding_(F_data)
        embeddings_S = self._GetEmbedding_(S_data)
        sim = {'s':[l for l in S_data]}
        k=0
        # Compute the sim of each F embedding and S embedding  
        for em in embeddings_F:
            cos_scores = self._GetSim_cos_(em, embeddings_S)
            # 通过参考表进行人工检测输出结果检验，考虑到现有方法依赖超参数，因此暂不使用
            #self._table_check_(F_data[k],S_data[k],sim)
            
            #softmax计算输出
            softmax_sim = np.exp(cos_scores.numpy())/np.sum(np.exp(cos_scores.numpy()))
            if type: #选择是否需要进行相似度的归一化计算
                hc_sim = (softmax_sim-softmax_sim.min())/(softmax_sim.max()-softmax_sim.min())
                sim[F_data[k]] = hc_sim.tolist()
            else:
                sim[F_data[k]] = cos_scores.numpy().tolist()
            
            k+=1
        if not os.path.exists(self.config.ResultsFile):
            os.mkdir(self.config.ResultsFile)
        # Save sim_matrix and ordered name into Resutls
        pd.DataFrame(sim).to_csv(self.config.ResultsFile + '/'+savefilename,
                                 encoding = "utf_8_sig")
        return sim,embeddings_F,embeddings_S

    def Select_Scenario(self,F_data,SS_data,sim,k1,k2,b,savefilename):
        # 调用上述人工检查方法进一步提升选出scenario的可靠性
        order1 = _Get_searched_Scenario_(F_data,self.config.ref_data)
        order2 = _Get_matched_Scenario_(F_data,self.config.ref_data)
        order3 = _Get_BM25_Scenario(k1,k2,b,F_data,SS_data)
        order4 = _Get_selected_Scenario_(sim,F_data,SS_data)
        order = order1.copy()
        # 按照给定流程将所有筛选的scenario与模型计算最契合的scenario合并并去重作为最终输出的scenario
        for k in order1.keys():
            order[k]=''
            for it in list(set(order1[k] + order2[k] + order3[k] + order4[k])):
                order[k] += it
                order[k] += ';'
        # 筛选结果保存为txt
        pd.DataFrame(order,index=[0]).to_csv(self.config.ResultsFile + '/' + savefilename, encoding = "utf_8_sig")
        return order

    def Fine_tune(self,Feature_list,Scenario_list,Feature_eval,Scenario_eval,epoch):
        # Prepare data
        train_data,sentences1,sentences2,scores = utl.training_data_SF(Feature_list,Scenario_list,
                                                                        Feature_eval,Scenario_eval,
                                                                        self.config.ref_data)
        # Prepare evaluator
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
        # Prepare finetune dataset
        train_dataset = SentencesDataset(train_data, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=100)
        # Set loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        #start training
        if not os.path.exists(self.config.finetuneFile + '/' +
                              self.config.model_name):
            os.mkdir(self.config.finetuneFile +  '/' +
                     self.config.model_name)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], 
                                    epochs=epoch, 
                                    warmup_steps=2, 
                                    evaluator=evaluator, 
                                    evaluation_steps=10, 
                                    output_path=self.config.finetuneFile +  '/' +
                                                self.config.model_name)
        eval = self._finetune_eval_(self.config.finetuneFile +  '/' +
                                    self.config.model_name + '/eval/similarity_evaluation_results.csv')
        return train_loss,evaluator,eval
    

    def Pre_train(self,train_sentences,dev_sentences,pretrain_config,save_path):
        # 预训练方法
        # 获取用于预训练的模型，可以项目中已经经过预训练的模型
        model = AutoModelForMaskedLM.from_pretrained(self.config.pretrainFile + self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.pretrainFile + self.config.model_name)
        # 准备训练参数、数据集、训练接口
        train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, pretrain_config.max_length)
        dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, pretrain_config.max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=pretrain_config.mlm_prob)
        training_args = TrainingArguments(
                                        output_dir= save_path,
                                        overwrite_output_dir=True,
                                        num_train_epochs=pretrain_config.num_train_epochs,
                                        evaluation_strategy="steps" if dev_dataset is not None else "no",
                                        per_device_train_batch_size=pretrain_config.per_device_train_batch_size,
                                        eval_steps=pretrain_config.save_steps,
                                        save_steps=pretrain_config.save_steps,
                                        logging_steps=pretrain_config.save_steps,
                                        save_total_limit=1,
                                        prediction_loss_only=True,
                                        fp16=pretrain_config.use_fp16
                                    )
        trainer = Trainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=dev_dataset
                        )
        # 开始训练
        trainer.train()

        # 保存预训练模型到pretrain文件夹，替换原有模型
        model.save_pretrained(self.config.pretrainFile + self.config.model_name)
        tokenizer.save_pretrained(self.config.pretrainFile + self.config.model_name)

        
             