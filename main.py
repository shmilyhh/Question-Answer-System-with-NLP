from BigramChunker import *
from LanguageProcessor import *

def main():
    
    print "Processing..." 
    
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Question & Answering Question')
    parser.add_argument("-t", "--text", help="the name of the text file")
    parser.add_argument("-q", "--question", help="the name of the question file")
    parser.add_argument("-i", "--interactive", help="the interactive model", action='store_true')
    
    args = parser.parse_args()
    text = args.text
    question = args.question
    
    # Bigram chunker
    # get the train data and text data from the conll2000 corpus       
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    # get the bigram chunker
    bigram_chunker = BigramChunker(train_sents)


    # Text
    # get the text data from reading the file
    with open(text, 'r') as f:
        text = f.read()
    
    if question and args.interactive:
        print("Error input! See -h")
        return
        
    if (not question) and (not args.interactive):
        print("Error input! See -h")
        return
        
    if question:
        print "Searching..." 
        with open(question, 'r') as q:
            questions = q.read()
    
        
    processor = LanguageProcessor()

    # Address the text

    # sentence tokenization
    sents = processor.sent_tokenizer.tokenize(text)
    # word tokenization
    word_tokenized_sents = processor.word_token(sents)
    # get the pos tag
    pos_tag_sents = processor.pos_tag(word_tokenized_sents)
    # get the name entity in the text
    name_entity_in_text = processor.get_ne(pos_tag_sents, sents, bigram_chunker, 'T')
    #print name_entity_in_text
    # get the name entity of the text in plain    
    plain_ne_in_text = processor.get_plain_ne_in_text(name_entity_in_text) 
    # get the map between name entity and index of the sentence that includes the name entity
    index_ne_to_sent_text = processor.set_index_ne_to_sent(name_entity_in_text)
    #print index_ne_to_sent_text
        
    while(True):
        if args.interactive:
            questions = raw_input('Enter the question: ')
            if questions=='q':
                print "Quiting..."
                break
            print "Searching..."    
            
        # Question
        # Address the question
    
        # sentence tokenization
        question_sents = processor.sent_tokenizer.tokenize(questions)
        # word tokenization
        question_word_tokenized_sents = processor.word_token(question_sents)
        # get the pos tag
        question_pos_tag_sents = processor.pos_tag(question_word_tokenized_sents)
        # get the name entity in the text
        question_name_entity = processor.get_ne(question_pos_tag_sents, question_sents, bigram_chunker, 'Q')
        # address the question name entity that not found by the ne_chunk
        processor.address_q_ne(question_name_entity, question_pos_tag_sents)

        # Construct the dictionary of the question
        pos = 0
        question_dict = {}
        for q in question_word_tokenized_sents:
            question_dict[pos] = {}
            question_dict[pos]['key_word'] = question_name_entity[pos][0]
            #print(q[0])
            if q[0] in ['did', 'do', 'does', 'Did', 'Does', 'Do']:
                question_dict[pos]['constrainted_word'] = re.findall(r'(\w+) or (\w+)', question_sents[pos])
                question_dict[pos]['ques_type'] = 'or'
                question_dict[pos]['ans_type'] = 'or'
            if q[0] in ['how', 'How', 'What', 'what']:
                cds = re.findall(r'(\w+) (\w+)\?', question_sents[pos])[0]
                constrainted_word = []
                indx = -3
                i = 0
                for cd in cds:
                    if question_pos_tag_sents[pos][indx+i][1] != 'NNP' and question_pos_tag_sents[pos][indx+i][1] != 'IN':
                        constrainted_word.append(question_pos_tag_sents[pos][indx+i][0])
                    i += 1
                if len(constrainted_word) == 2:
                    if question_pos_tag_sents[pos][-3][1] == 'VB' and question_pos_tag_sents[pos][-2][1] == 'RP':
                        constrainted_word_new = '_'.join(constrainted_word)
                        del constrainted_word[:]
                        #constrainted_word.clear()
                        constrainted_word.append(constrainted_word_new)
                question_dict[pos]['constrainted_word'] = constrainted_word
                question_dict[pos]['ques_type'] = 'how'
                if q[1] == 'much' or q[0] in ['What', 'what']:
                    # 'CD' is the pos tag of the number
                    question_dict[pos]['ans_type'] = 'CD'
            pos += 1
        print(question_dict)

        #Extract the answer
    
        #print(name_entity_in_text)
        #print(plain_ne_in_text)
        answers_dict = {}
        for q_idx in range(len(question_dict)):
            #print("question", q_idx)
            q_key_word = question_dict[q_idx]['key_word']
            q_key_word_syn = processor.get_syn(q_key_word)
            #print(q_key_word_syn)    
            idxs_sent_with_q_kw = processor.search_key_word([q_key_word], plain_ne_in_text, index_ne_to_sent_text)
            #print(idxs_sent_with_q_kw)
            #for i in idxs_sent_with_q_kw:
                #print sents[i]
            q_constrainted_words = question_dict[q_idx]['constrainted_word']
    
            # 'or' question
            if question_dict[q_idx]['ans_type'] == 'or':
                answer_of_question = []
                sources = []
                index = []
                q_constrainted_word1_syn = processor.get_syn(q_constrainted_words[0][0])
                #print(q_constrainted_word1_syn)
                q_constrainted_word2_syn = processor.get_syn(q_constrainted_words[0][1])
                #print(q_constrainted_word2_syn)
                for idx_sent_with_q_kw in idxs_sent_with_q_kw:
                    #print(idx_sent_with_q_kw)
                    #print(word_tokenized_sents[idx_sent_with_q_kw])
                    answer_key_word_idx1 = processor.get_key_word_idx(word_tokenized_sents, q_key_word, idx_sent_with_q_kw)
                    if processor.search_answer(q_constrainted_word1_syn, word_tokenized_sents[idx_sent_with_q_kw], answer_key_word_idx1):
                        answer_of_question.append(q_constrainted_words[0][0])
                        sources.append(sents[idx_sent_with_q_kw])
                        index.append(idx_sent_with_q_kw)
                        #answer_of_question = q_constrainted_words[0][0]
                        #break
                    elif processor.search_answer(q_constrainted_word2_syn, word_tokenized_sents[idx_sent_with_q_kw], answer_key_word_idx1):
                        answer_of_question.append(q_constrainted_words[0][1])
                        sources.append(sents[idx_sent_with_q_kw])
                        index.append(idx_sent_with_q_kw)
                        #answer_of_question = q_constrainted_words[0][1]
                        #break
                    #else:
                        #answer_of_question.append('None')
                
                answers_dict[q_idx] = {}
                answers_dict[q_idx]['answer'] = answer_of_question
                if answers_dict[q_idx]['answer']:
                    answers_dict[q_idx]['source'] =  sources
                    answers_dict[q_idx]['index'] =  index
                    #answers_dict[q_idx]['source_index'] = idx_sent_with_q_kw
                else:
                    answers_dict[q_idx]['source'] = None
                    #answers_dict[q_idx]['source_index'] =  None

            # 'how much' question
            if question_dict[q_idx]['ans_type'] == 'CD':
                answer_of_question2 = []
                sources = []
                index = []
                q_constrainted_word_syn = processor.get_syn(q_constrainted_words[0])
                #print(q_constrainted_word_syn)
                for idx_sent_with_q_kw in idxs_sent_with_q_kw:
                    print(word_tokenized_sents[idx_sent_with_q_kw])
                    answer_key_word_idx2 = processor.get_key_word_idx(word_tokenized_sents, q_key_word, idx_sent_with_q_kw)
                    if q_key_word == 'S&P':
                        answer_key_word_idx2 = 0
                    if processor.search_answer(q_constrainted_word_syn, word_tokenized_sents[idx_sent_with_q_kw], answer_key_word_idx2):
                        idx_wd = processor.search_answer(q_constrainted_word_syn, word_tokenized_sents[idx_sent_with_q_kw], answer_key_word_idx2)
                        #idx_wd = 0
                        #print(pos_tag_sents[idx_sent_with_q_kw])
                        anwser_scope = pos_tag_sents[idx_sent_with_q_kw][idx_wd-5:]
                        find_CD = False
                        for w_pos_tuple in anwser_scope:
                            if w_pos_tuple[1] == 'CD' or (w_pos_tuple[1] == 'TO' and find_CD):
                                find_CD = True
                                answer_of_question2.append(w_pos_tuple[0])
                        if answer_of_question2 and find_CD:
                            sources.append(sents[idx_sent_with_q_kw])  
                            index.append(idx_sent_with_q_kw)
                                  
                answers_dict[q_idx] = {}
                answers_dict[q_idx]['answer'] = answer_of_question2
                if answers_dict[q_idx]['answer']:
                    answers_dict[q_idx]['source'] = sources
                    answers_dict[q_idx]['index'] = index
                else:
                    answers_dict[q_idx]['source'] = None
                    answers_dict[q_idx]['index'] =  None    
        #print(answers_dict)                
        processor.address_answer(question_sents, answers_dict)
        
        if not args.interactive:
            break

main()