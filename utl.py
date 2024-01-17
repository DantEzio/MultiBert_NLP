### some tool

from copy import deepcopy
from random import randint,choice
from sentence_transformers import SentenceTransformer, util, losses, InputExample, SentencesDataset, evaluation

def shuffle(Scenario,S_data):
    # 用于构建负样本数据
    # 找现有feature不对应的scenario构建新的数据对，直接作为负样本，因此仅替换scenario对应关系
    # S_data:所有的scenario；Scenario：需要打乱顺序的scenario数据
    all_scenario = list(set(S_data))
    shuffle_Scenario = Scenario.copy()
    for i in range(len(shuffle_Scenario)):
        ind = all_scenario.index(shuffle_Scenario[i])
        snum = [i for i in range(len(all_scenario))]
        snum.pop(ind)
        shuffle_Scenario[i]=all_scenario[choice(snum)]
    return shuffle_Scenario

def combine_traindata(Feature,shuffle_Feature,Scenario,shuffle_Scenario,ref_data):
    # 构建微调的训练数据
    train_data = []
    for idx in range(len(Feature)):
        train_data.append(InputExample(texts=[Feature[idx], Scenario[idx]], label=1.0))
        
        # 负样本通过检验判断是否保留，检验方法：查询当前打乱顺序后数据对是否存在于已有列表中，如果有则不保留，否则作为eval data保存
        flag_train = True
        # 用于检验负样本的参考数据
        human_link = ref_data.copy()[:,0].tolist()
        human_link_s = ref_data.copy()[:,1].tolist() 
        while shuffle_Feature[idx] in human_link: # 逐一匹配，找到人工对应表的S文本
            # 找到正样本中对应数据对
            ind = human_link.index(shuffle_Feature[idx])
            stext = human_link_s[ind]
            # 如果打乱数据对出现在正样本中，则不作为负样本
            if stext == shuffle_Scenario[idx]:
                flag_train = False
                break
            human_link.pop(ind)
            human_link_s.pop(ind)
        if flag_train:
            train_data.append(InputExample(texts=[shuffle_Feature[idx], shuffle_Scenario[idx]], label=0.0))
    
    return train_data


def combine_evaldata(Feature,shuffle_Feature,Scenario,shuffle_Scenario,ref_data):
    # 构建微调的验证数据
    # 正样本保留
    sentences1 = Feature[:]
    sentences2 = Scenario[:]

    neg_k = 0 # 记录负样本个数
    for idx in range(len(shuffle_Feature)):
        # 负样本通过检验判断是否保留，检验方法：查询当前打乱顺序后数据对是否存在于已有列表中，如果有则不保留，否则作为eval data保存
        flag_eval = True
        # 用于检验负样本的参考数据
        human_link = ref_data.copy()[:,0].tolist()
        human_link_s = ref_data.copy()[:,1].tolist() 
        while shuffle_Feature[idx] in human_link: # 逐一匹配，找到人工对应表的S文本
            # 找到正样本中对应数据对
            ind = human_link.index(shuffle_Feature[idx])
            stext = human_link_s[ind]
            # 如果打乱数据对出现在正样本中，则不作为负样本
            if stext == shuffle_Scenario[idx]:
                flag_eval = False
                break
            human_link.pop(ind)
            human_link_s.pop(ind)
        if flag_eval:
            sentences1.append(shuffle_Feature[idx])
            sentences2.append(shuffle_Scenario[idx])
            neg_k+=1

    scores = [1.0] * len(Feature) + [0.0] * neg_k
    return sentences1,sentences2,scores


def training_data_SF(Feature_list,Scenario_list,Feature_eval,Scenario_eval,ref_data):
    # 本函数用于准备微调所需要的数据,train and eval use different data
    S_data = list(set(ref_data.copy()[:,1].tolist()))
    shuffle_Feature_list = Feature_list.copy()
    shuffle_Scenario_list = shuffle(Scenario_list,S_data) #　打乱排序
    shuffle_Feature_eval = Feature_eval.copy()
    shuffle_Scenario_eval = shuffle(Scenario_eval,S_data) #　打乱排序
    # print(len(Feature_list),len(Scenario_list))
    # Define your training data, include both positive and negeative sample
    train_data = combine_traindata(Feature_list,shuffle_Feature_list,Scenario_list,shuffle_Scenario_list,ref_data)
    
    # Define your evaluation examples, include both positive and negeative sample
    sentences1,sentences2,scores = combine_evaldata(Feature_eval,shuffle_Feature_eval,Scenario_eval,shuffle_Scenario_eval,ref_data)
    return train_data,sentences1,sentences2,scores



def data_check(F_data,S_data,dic):
    # 检验数据中的异常字符并做处理
    def _word_check_(word,dic):
        # 对词语进行检验
        return word not in dic
    #对数据进行批量检验
    temf,tems=[],[]
    for f,s in zip(F_data,S_data):
        if (_word_check_(f,dic)):
            for te in dic:
                f.strip(te)
        if (_word_check_(s,dic)):
            for te in dic:
                s.strip(te)
        temf.append(f)
        tems.append(s)
    F_data,S_data=temf,tems
    return F_data,S_data

