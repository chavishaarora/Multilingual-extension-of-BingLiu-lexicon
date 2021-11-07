from gensim import utils

temp_sentences=open("./assignment_4_files/english.txt").read().splitlines()

clean_sentences_english=list()
for i in temp_sentences:
	clean_sentences_english.append(utils.simple_preprocess(i))

with open('tokenized_english.txt', 'w') as f:
    for item in clean_sentences_english:
        f.write("%s\n" % item)


temp_sentences=open("./assignment_4_files/hindi.txt").read().splitlines()

clean_sentences_hindi=list()
for i in temp_sentences:
	x=i.split()
	clean_sentences_hindi.append(x)

with open('tokenized_hindi.txt', 'w') as f:
    for item in clean_sentences_hindi:
        f.write("%s\n" % item)