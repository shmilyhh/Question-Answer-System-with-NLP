import nltk
from nltk.tokenize import word_tokenize
from nltk import chunk
import re
from nltk.corpus import brown
from nltk.corpus import conll2000
from nltk.corpus import wordnet as wn
from pattern.en import lemma
import argparse

# train the tagger by using the unigram and bigram
"""
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
"""


class LanguageProcessor(object):
    """
    provides some functions to process the natural language
    """
    def __init__(self):
        #self.text = text
        #self.question = question
        # load the sentence tokenizer
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    def word_token(self, tokenized_sent):
        """
        This function tokenizes the sentece by word.
        Parameters:
            tokenized_sent (list) - the element of this list is sentence of the text
        Variables:
            tokenized_sent (list) - the list of the words in one sentence
        return wd_tokenized_sents (list) - the list of the tokenized_sent
        """
        wd_tokenized_sents = []
        for sent in tokenized_sent:
            tokenized_sent = word_tokenize(sent)
            # in order to get the string like 'S&P'
            if re.findall(r'[A-Z]&[A-Z]', sent):
                ne_abbred = re.findall(r'[A-Z]&[A-Z]', sent)
                tokenized_sent += ne_abbred
                wd_tokenized_sents.append(tokenized_sent)
            #tokenized_sent = nltk.regexp_tokenize(sent, pattern)
            else:
                wd_tokenized_sents.append(tokenized_sent)
        return wd_tokenized_sents
    
    def pos_tag(self, wd_tk_sens):
        """
        This function gets the pos tag of each word in the sentece
        Parameters:
            wd_tk_sens (list) - a list of the sentences with word tokenization
        Variables:
            pt_sent (list) - a list of the tuple (word, pos_tag)
        return pt_sents (list) - a list of the pt_sent
        """
        pt_sents = []
        for word_tokenized_sent in wd_tk_sens: 
            pt_sent = nltk.pos_tag(word_tokenized_sent)
            #pos_tag_sent = t2.tag(word_tokenized_sent)
            pt_sents.append(pt_sent)
        return pt_sents
    
    def traverse(self, t, nel):
        """
        This function traverse the name entity tree recurively, and extract the leaves of the NE
        Parameters:
            t (tree) - the name entity tree
            nel (list) - the name entity list, to store the name entities in the tree
        return nel
        """
        try:
            t.label()
        except AttributeError:
            pass
            #print(t, end='')
        else:
            # ne_chunk: the root of the ne is 'NE' and BigramChunker: 'NP'
            if t.label() == 'NP' or t.label() == 'NE':
                nel.append(t.leaves())
            #print("(", t.label(), end='')
            for child in t:
                self.traverse(child, nel)
            #print(")", end='')
        return nel
    
    def get_name_entity_list(self, tree):
        """
        This function gets the each word in the name entity, stores them in the list
        Parameters:
           tree (tree) - the name entity tree 
        Variables:
            name_entity_list (list) - the list of the ne and the element of the list 
                                      is the list of tuple (word, pos_tag)
        return name_entity_list
        """
        name_entity_list = []
  
        return self.traverse(tree, name_entity_list)

    def get_name_entity(self, name_entity_list):
        """
        This function addresses the each word in one name entity to get them into word phrase
        Parameters:
            name_entity_list (list) - the list of the ne and the element of the list 
                                      is the list of tuple (word, pos_tag)
        Variables:
            name_entity_item (list) - the list of the words in one name entity
            name_entity_sent (list) - the list of the name entities in one name entity tree
        return name_entity_sent
        """
        name_entity_sent = []
        for ne in name_entity_list:
            name_entity_item = []
            for tup in ne:
                name_entity_item.append(tup[0])
            name_entity_sent.append(' '.join(name_entity_item))
      
        return name_entity_sent
      
    def get_ne(self, post_taged_sents, sent_tokenized_sents, chunker, mark):
        """
        This function gets all the name entity in the text.
        Parameters:
            post_taged_sents (list) - pos taged sentece
            sent_tokenized_sents (list) - sentenced tokenized text
            mark (string) - 'Q' means question; 'T' means text. To decide which chunker is used
        Variables:
            name_entity_sents (list) - the list of all the name entities in the text, 
                                       the element of the list is the list of the ne 
                                       of each sentence
        return name_entity_sents
        """
        name_entity_sents = []
        idx = 0
        for pos_tag_sent in post_taged_sents:
            #print(len(post_taged_sents))
            #print()
            #print(pos_tag_sent)
            #print(idx)
            if re.findall(r'[A-Z]&[A-Z]', sent_tokenized_sents[idx]):
                ne_abbred = re.findall(r'[A-Z]&[A-Z]', sent_tokenized_sents[idx])
                #name_entity_sents.append(ne_abbred)
            else:
                ne_abbred = []
            if mark == 'Q':
                ne_chunk = chunk.ne_chunk(pos_tag_sent, binary=True)
                #name_entity_sents.append(get_name_entity(get_name_entity_list(ne_chunk)))
            if mark == 'T':
                ne_chunk = chunker.parse(pos_tag_sent)
                #name_entity_sents.append(get_name_entity(get_name_entity_list(ne_chunk)))
            #print(ne_chunk)
            ne_chunks = ne_abbred + self.get_name_entity(self.get_name_entity_list(ne_chunk))
            name_entity_sents.append(ne_chunks)
            idx += 1
        return name_entity_sents

    def get_plain_ne_in_text(self, name_entities):
        """
        This function combines the total name entity into one list
        Parameters:
            name_entities (list) - the list of all the name entities in the text, 
                                   the element of the list is the list of the ne 
                                   of each sentence
  
        Variables:
            plain_ne_in_text (list) - the list of the total name entities
        return  plain_ne_in_text
        """
        plain_ne_in_text = []
        for ne_list in name_entities:
            if ne_list:
                for ne in ne_list:
                    if ne not in plain_ne_in_text:
                        plain_ne_in_text.append(ne)
        return plain_ne_in_text

    #Index
    def set_index_ne_to_sent(self, name_entities):
        """
        This function sets the name entity to the index of the sentence that includes the name entity
        Parameters:
            name_entities (list) - the list of the name entities
        Variables:
            index_ne_to_sent (dict) - the map between name entity(key) and the index(value) of the sentence
                                      the value of the dictionary is the list
        return index_ne_to_sent
        """
        index_ne_to_sent = {}
        idx = 0
        for ne_list in name_entities:
            if ne_list:
                for ne in ne_list:
                    if ne in index_ne_to_sent.keys():
                        index_ne_to_sent[ne].append(idx)
                    else:
                        index_ne_to_sent[ne] = [idx]
            idx += 1
        return index_ne_to_sent
        
    # Question   
    def address_q_ne(self, q_ne, q_pt_sents):
        """
        This function addresses the question name entity that not found by the ne_chunk
        Parameters:
            q_ne (list) - the list of the name entities of the quesions,
                          the element of the list is the list of the ne of each question
        """
        for m, n in enumerate(q_ne):
            if not n:
                previous = False
                #print "m:", m
                for t in q_pt_sents[m]:
                    if t[1] == 'NNP': #or (t[1] == 'CC' and previous):
                        previous = True
                        n.append(t[0])

    def get_syn(self, word):
        """
        This function gets the synonyms of the word using WordNet
        Parameters:
            word (string) - the word
        Variables:
            synsets_list (list) - the list of the synonyms
        return synsets_list
        """
        synsets_list = []
        if word == 'fall' or word == 'drop' or word == 'down' :
            synsets_list.append('lower')
            synsets_list.append('down')
        if wn.synsets(word):
            for synset in wn.synsets(word):
                for lemma_name in synset.lemma_names():
                    if lemma_name not in synsets_list:
                        synsets_list.append(lemma_name)
        else:
            synsets_list.append(word)
        return synsets_list

    def search_key_word(self, key_word_syn, name_entities_text, indexes_ne_to_sent_text):
        """
        This function uses the key word in the question to match the key word in the text
        and get the sentences index of the key word in the text
        Parameters:
            key_word_syn (list) - the list of the synonmys
            name_entities_text (list) - the list of the ne in the text
            indexes_ne_to_sent_text (dict) - the map between ne and sentences index
        Variables:
            idx_sent_list (list) - the list of the index of the sentences that contain the question key word
        return idx_sent_list
        """
        idx_sent_list = []
        #print(key_word_syn):
        #print(name_entities_text)
        for qw in key_word_syn:
            for ne_text in name_entities_text:
                #print(qw)
                #print(ne_text)
                if qw in ne_text or ne_text in qw:
                    #print qw, ne_text
                    for ind in indexes_ne_to_sent_text[ne_text]:
                        #print ind
                        if ind not in idx_sent_list:
                            idx_sent_list.append(ind)
                    
        return idx_sent_list

    def get_key_word_idx(self, wd_tokenized_sents, key_word, idx_sent_list):
        """
        This function gets the position of the key word in the sentence
        Parameters:
            wd_tokenized_sents (list) - word tokenization text
            key_word (string) - word
            idx_sent_list (list) - the list of the index of the sentences that contain the key word
        Varibales:
            key_word_idx (int) - the position of the key word in the sentence
        """
        #print(idx_sent_list, key_word)
        if key_word in wd_tokenized_sents[idx_sent_list]:
            key_word_idx = wd_tokenized_sents[idx_sent_list].index(key_word)
        else:
            key_word_idx = 0
        return key_word_idx

    def search_answer(self, cnstrd_word_syn, wd_in_sent, key_wd_idx):
        """
        This function searches the constrainted word of the question
        Parameters:
            cnstrd_word_syn (list) - the list of the synonyms of the constrainted word in the question
            wd_in_sent (list) - word tokenization text
            key_wd_idx (int) - the position of the key word in the sentence
        return: the position of the constrainted word in the sentence of the text  
        """
        porter = nltk.PorterStemmer()
        lancaster = nltk.LancasterStemmer()
        #print cnstrd_word_syn
        for cw in cnstrd_word_syn:
            cw_seperate = []
            if '_' in cw:
                
                cw1 = cw.split('_')[0]
                cw2 = cw.split('_')[1]
                cw_seperate = [cw1, cw2]
                
                cw = ' '.join(cw.split('_'))
                
                cw_seperate.append(cw)
            #print(cw)
            for sent in wd_in_sent[key_wd_idx:]:
                #print(cw)
                #print(cw, sent)
                #print sent
                """
                if cw_seperate:
                    for c_s in cw_seperate:
                        if porter.stem(c_s.lower()) == porter.stem(sent.lower()) or lemma(c_s) == lemma(sent): #or sent.lower() in cw.lower() or lemma():
                            print("!!!!!!!!")
                            print(cw, sent)
                            print(wd_in_sent.index(sent))
                            return wd_in_sent.index(sent)
                """
                if porter.stem(cw.lower()) == porter.stem(sent.lower()) or lemma(cw) == lemma(sent): #or sent.lower() in cw.lower() or lemma():
                    #print("!!!!!!!!")
                    #print(cw, sent)
                    #print(wd_in_sent.index(sent))
                    return wd_in_sent.index(sent)
                """
                elif cw_seperate:
                    for cw_s in cw_seperate:
                        if porter.stem(cw.lower()) == porter.stem(sent.lower()) or lemma(cw) == lemma(sent):
                            return wd_in_sent.index(sent)
                """     
        return None

    def address_answer(self, questions, answers):
        """
        This function addresses the format of the answers of the questions, and print the answer
        Parameters:
            answers (list) - the list of the answers
        Variables:
            answer (string) - the answer
        """
        for k in answers.keys():
            #if type(answers[k]['answer']) is list:
            if answers[k]['answer']:
                answer = answers[k]['answer']
                source = answers[k]['source']
                index = answers[k]['index']
            else:
                answer = 'Not found!'
            #else:
                #answer = answers[k]['answer']
        
            print "Q: ", questions[k]
            print "A: ", answer
            if answer != 'Not found!':
                for seq in range(len(answers[k]['source'])):
                    print "A: ", answer[seq] 
                    print "Source: ", source[seq] #("(%dth sentence)" %answers[k]['source_index'])
                    print "Index: ", index[seq]
                    
            print          