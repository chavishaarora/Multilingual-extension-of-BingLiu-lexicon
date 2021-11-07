import pandas as pd
from gensim import utils
import gensim.models
import numpy as np
from scipy import spatial

bingliu_df=pd.read_csv("./assignment_4_files/BingLiu.csv",sep="\t",header=None)
print("Bingliu-\n",bingliu_df)

en_hi_dict=dict()
temp_sentences=open("./assignment_4_files/english-hindi-dictionary.txt").read().splitlines()
for k, word in enumerate(temp_sentences):
	x=word.split()
	if len(x)>=3:
		if x[0].lower()==x[2].lower():
			# print(x)
			continue
		else:
			en_hi_dict[x[0]]=x[2]

print("Number of objects in english-hindi-dictionary- ",len(en_hi_dict))

L1=pd.DataFrame(columns=["English","Hindi","Polarity"])
bingliu_dict=dict()
for ind in bingliu_df.index:
	bingliu_dict[bingliu_df[0][ind]]=bingliu_df[1][ind]

counter=0
for key in en_hi_dict:
	if key in bingliu_dict:
		L1.loc[counter]=[key]+[en_hi_dict[key]]+[bingliu_dict[key]]
		counter+=1	

print("-"*20, " ORIGINAL L1 ", "-"*20)
print(L1)


# ----------- Training Word2Vec model on english.txt ------------
# temp_sentences=open("./assignment_4_files/english.txt").read().splitlines()
# clean_sentences_english=list()
# for i in temp_sentences:
# 	clean_sentences_english.append(utils.simple_preprocess(i))
# 	# words=i.lower().split()
# 	# clean_sentences_english.append(words)
# model_english=gensim.models.Word2Vec(sentences=clean_sentences_english)

# # print(model_english.wv.most_similar(positive=["good"],topn=6))

# # ----------- Training Word2Vec model on hindi.txt ------------
# sentences_hindi=open("./assignment_4_files/hindi.txt").read().splitlines()
# clean_sentences_hindi=[]
# for i in sentences_hindi:
# 	words=i.split()
# 	clean_sentences_hindi.append(words)
# model_hindi=gensim.models.Word2Vec(sentences=clean_sentences_hindi)

#I Importing the saved word2vec models
model_english=gensim.models.Word2Vec.load("word2vec_english.model")
model_hindi=gensim.models.Word2Vec.load("word2vec_hindi.model")


def check_combinations(eng_5,hindi_5):
	valid_combinations=list()	
	for eng_word in eng_5:
		if eng_word in en_hi_dict:
			# print('-'*20,"YESSSS",'-'*20)
			hindi_word=en_hi_dict[eng_word]	
			if hindi_word in hindi_5:
				# print('-'*40,"HINDIIII ALSOOOO",'-'*40)
				valid_combinations.append([eng_word, hindi_word])
	return valid_combinations

# print(model_hindi.wv.most_similar(positive=["सही"],topn=6))

embeddings_dict = {}
with open("./vectors_english.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        if word[0]=='[':
        	embeddings_dict[word[2:-2]] = vector
        else:
        	embeddings_dict[word[1:-2]] = vector

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))        


embeddings_dict_hindi = {}
with open("./vectors_hindi.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        if word[0]=='[':
        	embeddings_dict_hindi[word[2:-2]] = vector
        else:
        	embeddings_dict_hindi[word[1:-2]] = vector

def find_closest_embeddings_hindi(embedding):
    return sorted(embeddings_dict_hindi.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict_hindi[word], embedding))        

already_exist=dict()
for ind in L1.index:
	already_exist[L1["English"][ind]]=L1["Hindi"][ind]

total_added=0
for k in range(4):
	print("\n")
	print("-"*40, " ITERATION NUMBER- ",k+1, "-"*40)
	added_w2v=0
	for ind in L1.index:
		eng_5=[]
		hindi_5=[]
		try:
			eng_5=model_english.wv.most_similar(positive=[L1["English"][ind]],topn=5)		
		except:		
			continue
		try:
			hindi_5=model_hindi.wv.most_similar(positive=[L1["Hindi"][ind]],topn=5)
		except:		
			continue
		if len(eng_5)>0 and len(hindi_5)>0:			
			eng_5_temp=list()
			for i in eng_5:
				i=list(i)
				eng_5_temp.append(i[0])
			eng_5=eng_5_temp
			hindi_5_temp=list()
			for i in hindi_5:
				i=list(i)
				hindi_5_temp.append(i[0])
			hindi_5=hindi_5_temp
			valid_combinations=check_combinations(eng_5,hindi_5)
			for i in range(len(valid_combinations)):				
				if valid_combinations[i][0] not in already_exist:
					L1.loc[counter]=[valid_combinations[i][0]]+[valid_combinations[i][1]]+[L1["Polarity"][ind]]
					counter+=1
					already_exist[valid_combinations[i][0]]=valid_combinations[i][1]
					added_w2v+=1
	print("\nAfter Word2Vec, iteration number-",k+1," | Added- ",added_w2v)
	print(L1)
	added_glove=0
	for ind in L1.index:
		eng_5=[]
		hindi_5=[]
		try:
			eng_5=find_closest_embeddings(embeddings_dict[L1["English"][ind]])[1:6]			
		except:		
			continue
		try:
			hindi_5=find_closest_embeddings_hindi(embeddings_dict_hindi[L1["Hindi"][ind]])[1:6]			
		except:		
			continue
		if len(eng_5)>0 and len(hindi_5)>0:					
			valid_combinations=check_combinations(eng_5,hindi_5)
			for i in range(len(valid_combinations)):				
				if valid_combinations[i][0] not in already_exist:
					L1.loc[counter]=[valid_combinations[i][0]]+[valid_combinations[i][1]]+[L1["Polarity"][ind]]
					counter+=1
					already_exist[valid_combinations[i][0]]=valid_combinations[i][1]
					added_glove+=1
	print("\nAfter Glove, iteration number-",k+1," | Added- ",added_glove)
	print(L1)
	total_added=total_added+added_w2v+added_glove

print("\n\nTotal new additions to L1 are- ",total_added,"\n\n")
