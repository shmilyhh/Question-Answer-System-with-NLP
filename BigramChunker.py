import nltk
class BigramChunker(nltk.ChunkParserI):
    """
    This class, inheritated from nltk.ChunkParserI,
    is about the name entity chunker, using the BigramTagger.
    It assigns each word IOB tag based on the pos_tag.
    function:
        __init__ - initialize the train data by getting the
                   tag and chunk IOB tag of each word.
                   And train the Bigramtagger by using the train data.
        parse    - get the IOB tag tree of the sentece.
    """
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)
    
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags] 
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)] 
        return nltk.chunk.conlltags2tree(conlltags)
