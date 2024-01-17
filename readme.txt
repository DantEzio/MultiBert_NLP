## NLP_sim

- 中文文本语义相似度计算模块。计算得到两个文本的相似度，并最终计算该feature与所有scenario的相似度，并结合人工检验方法构建的多路召回与多路输出，筛选与该feature语义接近的所有scenario。
- 借助Sentence Transformers与Pytorch构建。使用过程首先通过具体案例数据对已经完成预训练的模型进行微调，之后使用微调后模型进行语义量化特征计算与相似度（贡献度）计算。采用的预训练模型均为Bert类模型，包括bert-base-chinese、all-MiniLM-L6-v2、ms-marco-MiniLM-L-6-v2、multi-qa-MiniLM-L6-cos-v1、paraphrase-MiniLM-L6-v2，最终选择在案例词库上性能最佳的模型作为预训练模型应用。已经完成预训练的上述模型均来自hugging face的开源模型库。主要功能包括：文本词嵌入向量计算功能、词嵌入向量相似性（贡献度）分析功能、基于相似度的两组文本之间贡献度矩阵计算功能、语言模型微调功能、语言模型预训练功能。每个部分通过相应Python方法接口设计实现。

## 环境依赖
模块通过Python编写，Python版本推荐为3.7.9。具体使用的相关package信息详见requirement.txt。
numpy==1.26.3
pandas==2.1.4
sentence_transformers==2.2.2
torch==2.1.2
torch_xla==2.1.0
tqdm==4.66.1
transformers==4.36.2

## 文件描述
- config.py	//模型构建参数
- NLP_model.py	//模型类与多路召回人工检验功能
- utl.py	//使用工具
- requirements.txt	//安装package
- demo.ipynb	//应用案例
- data	//相关人工检验、微调、预训练数据
	-- Dataset	//补充收集的预训练使用数据，原始数据
	-- pretrain_testdata2.csv	//补充收集的预训练使用测试集
	-- pretrain_testdata.csv		//补充收集的预训练使用测试集
	-- pretrain_traindata.csv	//补充收集的预训练使用训练集
	-- pretrain_traindata2.csv	//补充收集的预训练使用训练集
	-- Book2.xlsx		//大众提供feature与scenario数据
	-- S_data.xlsx		//大众提供整理的scenario数据，作为测试集
	-- s2f_v2.xlsx		//大众提供feature与scenario数据，作为训练集和人工检验参考数据
- finetune_model	//保存微调后的模型文件夹
	-- all-MiniLM-L6-v2	//模型all-MiniLM-L6-v2
	-- bert-base-chinese	//新增模型bert-base-chinese
	-- ms-marco-MiniLM-L-6-v2	//模型ms-marco-MiniLM-L-6-v2
	-- multi-qa-MiniLM-L6-cos-v1	//模型multi-qa-MiniLM-L6-cos-v1
	-- paraphrase-MiniLM-L6-v2	//模型paraphrase-MiniLM-L6-v2
- pretrain_model	//保存下载的已经经过预训练的开源模型
	-- all-MiniLM-L6-v2	//模型all-MiniLM-L6-v2
	-- bert-base-chinese	//新增模型bert-base-chinese
	-- ms-marco-MiniLM-L-6-v2	//模型ms-marco-MiniLM-L-6-v2
	-- multi-qa-MiniLM-L6-cos-v1	//模型multi-qa-MiniLM-L6-cos-v1
	-- paraphrase-MiniLM-L6-v2	//模型paraphrase-MiniLM-L6-v2
- Results	//保存模型计算结果文件夹

## 模型相关
- bert预训练模型放在 pretain_model目录下，均为hugging face上提供的开源模型，且已经过预训练。
- 预训练模型下载地址：  
all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
bert-base-chinese: https://huggingface.co/bert-base-chinese
ms-marco-MiniLM-L-6-v2: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
multi-qa-MiniLM-L6-cos-v1: https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
paraphrase-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2

- 模型均经过微调数据微调，微调后模型放在finetune_model目录下的同名文件
- 下载后，放在对应目录下，文件名称确认无误即可。  


## 接口与类描述
- class config: 
模型构建参数，调用方法如下ct = config.config(finetuneFile='./finetune_model/',  #微调模型保存路径
                    pretrainFile = './pretrain_model/', #预训练模型保存路径
                    resultsFile = './Results/', #结果保存路径
                    initial = True, #用于选取读取预训练模型还是微调模型，False为读取微调模型
                    model_name = model_name, #选取模型名字，目前准备模型有如上注释
                    max_expand = 1.5, #人工检验中用于扩大人工选择匹配文本对相似度的扩大倍数，该功能暂时禁用，因此采用默认值输入
                    ref_data = ref_data)

- class pretrain_config:
预训练过程使用参数，调用方法如下pre_ct = config.pretrain_config(per_device_train_batch_size=64,  #batch_size
                    save_steps = 1, #Save model every given steps
                    num_train_epochs = 1, #Number of epochs
                    use_fp16 = False, #Set to True, if your GPU supports FP16 operations
                    max_length = 512, #Max length for a text input
                    do_whole_word_mask = True, #If set to true, whole words are masked
                    mlm_prob = 0.15 #Probability that a word is replaced by a [MASK] token)

- utl.py：模型调用相关方法
	-- utl.shuffle(Scenario,S_data): 
用于构建负样本数据的打乱数据步骤，找现有feature不对应的scenario构建新的数据对，直接作为负样本，因此仅替换scenario对应关系，S_data:所有的scenario；Scenario：需要打乱顺序的scenario数据

	-- utl.combine_traindata(Feature,shuffle_Feature,Scenario,shuffle_Scenario,ref_data): 
构建微调的训练数据

	-- utl.combine_evaldata(Feature,shuffle_Feature,Scenario,shuffle_Scenario,ref_data): 
构建微调的验证数据

	-- utl.training_data_SF(Feature_list,Scenario_list,Feature_eval,Scenario_eval,ref_data): 
本函数调用shuffle、combine_traindata、combine_evaldata用于准备微调所需要的数据,train and eval use different data。

	-- utl.data_check(F_data,S_data,dic): 检验数据中的异常字符并做处理

-NLP_sim：NLP模型类与多路召回功能，用于构建模型，初始化方法如下：Bert_model = NLP_model.NLP_sim(ct) #ct为设定的config
	-- NLP_sim._Get_selected_Scenario_(sim,F_data,S_data): 
模型输出中使用的人工检查多路召回功能，4个之一。使用模型计算的sim获取相似度最高的10个scenario，根据计算的相似度排序，保存所有scenario中前10个

	-- NLP_sim._Get_searched_Scenario_(F_data,ref_data): 
模型输出中使用的人工检查多路召回功能，4个之一。使用参考数据获取scenario，在给定ref_data中找到feature，并将数据中对应scenario返回保存

	-- NLP_sim._Get_matched_Scenario_(F_data,ref_data): 
模型输出中使用的人工检查多路召回功能，4个之一。统计与feature中含有相同文字的scenario，给定feature，将与feature中有相同文字的scenario返回保存，SS_data为所有的scenario保存的列表

	-- NLP_sim._Get_BM25_Scenario(k1,k2,b,F_data,SS_data): 
模型输出中使用的人工检查多路召回功能，4个之一。利用BM2.5模型筛选scenario，作为NLP模型辅助结果，参考https://zhuanlan.zhihu.com/p/79202151。给定feature，通过BM2.5将scenario中与feature相近的词语返回保存，每个f的每个词都可以对s统计一个词频，因此一个f含有多个词频，最终取词频最高的s作为输出。由于S作为文档太短，会出现dft为0的情况，因此对W做了修正，改为dft+1。SS_data为所有的scenario保存列表，k1 k2取1.2~2，b取值0.75，参考：https://zhuanlan.zhihu.com/p/79202151。

	-- NLP_sim._GetEmbedding_(words): 
计算文本的embedding

	-- NLP_sim._GetSim_cos_(em1,em2): 
计算两个embedding的相似度

	-- NLP_sim._finetune_eval_(model_link): 
Get the fine-tune curl

	-- NLP_sim._table_check_(F_data,S_data,sim): 
通过参考表进行人工检测输出结果检验，使用扩大因数，应要求已暂时禁用

	-- NLP_sim.Get_SimMatrix(type,F_data,S_data,savefilename): 
Input two sets of data to obtain sim_matrix，type表示是否需要进行相似度的归一化计算，sacefilename为保存结果路径

	-- NLP_sim.Select_Scenario(F_data,SS_data,sim,k1,k2,b,savefilename):
调用上述人工检查多路召回方法进一步提升选出scenario的可靠性。F_data和SS_data为模型的feature与所有可选scenario，sim为Get_SimMatrix计算相似度矩阵结果，k1、k2、b为BM2.5模型筛选所需参数，取值参考：https://zhuanlan.zhihu.com/p/79202151，savefilename为结果保存路径。

	-- NLP_sim.Fine_tune(Feature_list,Scenario_list,Feature_eval,Scenario_eval,epoch): 
微调模型，Feature_list,Scenario_list,Feature_eval,Scenario_eval分别为微调使用的训练与测试数据，epoch为迭代次数

	-- NLP_sim.Pre_train(train_sentences,dev_sentences,pretrain_config,save_path):
预训练方法。已有研究表明，预训练需要大量的数据与训练资源支持才有较好效果，而受限于行业公开数据资源的限制，这里的预训练代码效果可能不佳，在补充更多数据后可能有更好效果，因此补充的预训练代码模块仅作为参考保留。train_sentences是预训练过程使用训练数据，dev_sentences是预训练过程使用验证数据，pretrain_config是预训练过程相关参数，save_path是预训练过程保存文件夹路径，不能含有英文。

