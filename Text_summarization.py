import nltk
from nltk.corpus import stopwords
import csv
from nltk.tag import pos_tag # for proper noun
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import math
import re
ps = PorterStemmer()



caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

stop = set(stopwords.words('english'))

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    #if "," in text: text = text.replace(",\"","\",")

    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    #text = text.replace(",","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

#named entity recoginition
def ner(sample):
    sentences = nltk.sent_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    return len(entity_names)  

#Using Jaccard similarity to calculate if two sentences are similar
def similar(tokens_a, tokens_b) :
    ratio = len(set(tokens_a).intersection(tokens_b))/ float(len(set(tokens_a).union(tokens_b)))
    return ratio

# ......................part 1 (cue phrases).................
def cue_phrases(sent_tokens):
    QPhrases=["incidentally", "example", "anyway", "furthermore","according",
            "first", "second", "then", "now", "thus", "moreover", "therefore", "hence", "lastly", "finally", "summary"]

    cue_phrases={}
    for sentence in sent_tokens:
        cue_phrases[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for word in word_tokens:
            if word.lower() in QPhrases:
                cue_phrases[sentence] += 1
    maximum_frequency = max(cue_phrases.values())
    for k in cue_phrases.keys():
        try:
            cue_phrases[k] = cue_phrases[k] / maximum_frequency
            cue_phrases[k]=round(cue_phrases[k],3)
        except ZeroDivisionError:
            x=0
    print(cue_phrases.values())
    return cue_phrases


# .......................part2 (numerical data)...................
def numeric_data(sent_tokens):
    numeric_data={}
    for sentence in sent_tokens:
        numeric_data[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isdigit():
                numeric_data[sentence] += 1
    maximum_frequency = max(numeric_data.values())
    for k in numeric_data.keys():
        try:
            numeric_data[k] = (numeric_data[k]/maximum_frequency)
            numeric_data[k] = round(numeric_data[k], 3)
        except ZeroDivisionError:
            x=0
    print(numeric_data.values())
    return numeric_data


#....................part3(sentence length)........................
def sent_len_score(sent_tokens):
    sent_len_score={}
    for sentence in sent_tokens:
        sent_len_score[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        if len(word_tokens) in range(0,10):
            sent_len_score[sentence]=1-0.02*(10-len(word_tokens))
        elif len(word_tokens) in range(10,20):
            sent_len_score[sentence]=1
        else:
            sent_len_score[sentence]=1-(0.02)*(len(word_tokens)-20)
    for k in sent_len_score.keys():
        sent_len_score[k]=round(sent_len_score[k],4)
    print(sent_len_score.values())
    return sent_len_score


#....................part4(sentence position)........................
def sentence_position(sent_tokens):
    sentence_position={}
    d=1
    no_of_sent=len(sent_tokens)
    for i in range(no_of_sent):
        a=1/d
        b=1/(no_of_sent-d+1)
        sentence_position[sent_tokens[d-1]]=max(a,b)
        d=d+1
    for k in sentence_position.keys():
        sentence_position[k]=round(sentence_position[k],3)
    print(sentence_position.values())
    return sentence_position


#.........Create a frequency table to compute the frequency of each word........
def word_frequency(sent_tokens,word_tokens_refined):
    freqTable = {}
    for word in word_tokens_refined:    
        if word in freqTable:         
            freqTable[word] += 1    
        else:         
            freqTable[word] = 1
    for k in freqTable.keys():
        freqTable[k]= math.log10(1+freqTable[k])
#Compute word frequnecy score of each sentence
    word_frequency={}
    for sentence in sent_tokens:
        word_frequency[sentence]=0
        e=nltk.word_tokenize(sentence)
        f=[]
        for word in e:
            f.append(ps.stem(word))
        for word,freq in freqTable.items():
            if word in f:
                word_frequency[sentence]+=freq
    maximum=max(word_frequency.values())
    for key in word_frequency.keys():
        try:
            word_frequency[key]=word_frequency[key]/maximum
            word_frequency[key]=round(word_frequency[key],3)
        except ZeroDivisionError:
            x=0
    print(word_frequency.values())
    return word_frequency


#........................part 6 (upper cases).................................
def upper_case(sent_tokens):
    upper_case={}
    for sentence in sent_tokens:
        upper_case[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isupper():
                upper_case[sentence] += 1
    maximum_frequency = max(upper_case.values())
    for k in upper_case.keys():
        try:
            upper_case[k] = (upper_case[k]/maximum_frequency)
            upper_case[k] = round(upper_case[k], 3)
        except ZeroDivisionError:
            x=0
    print(upper_case.values())
    return upper_case


#......................... part7 (number of proper noun)...................
def proper_noun(sent_tokens):
    proper_noun={}
    for sentence in sent_tokens:
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
        proper_noun[sentence]=len(propernouns)
    maximum_frequency = max(proper_noun.values())
    for k in proper_noun.keys():
        try:
            proper_noun[k] = (proper_noun[k]/maximum_frequency)
            proper_noun[k] = round(proper_noun[k], 3)
        except ZeroDivisionError:
            x=0
    print(proper_noun.values())
    return proper_noun


#.................. part 8 (word matches with heading) ...................
def head_match(sent_tokens):
    head_match={}
    heading=sent_tokens[0]
    stopWords = list(set(stopwords.words("english")))
    for sentence in sent_tokens:
        head_match[sentence]=0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k not in stopWords:
                k = ps.stem(k)
                if k in ps.stem(heading):
                    head_match[sentence] += 1
    maximum_frequency = max(head_match.values())
    for k in head_match.keys():
        try:
            head_match[k] = (head_match[k]/maximum_frequency)
            head_match[k] = round(head_match[k], 3)
        except ZeroDivisionError:
            x=0
    print(head_match.values())
    return head_match


#..................... part 9(Centrality)..........................
def centrality(sent_tokens,cv,word_tokens_refined):
    global corpus
    l=len(sent_tokens)
    centrality_value={}
    Tf_Idf = cv.fit_transform(corpus).toarray()
    
    # Cosine Similarity
    cosine_value={}
    for i in range(0,l):
        sentence=sent_tokens[i]
        cosine_value[sentence]=0
        for j in range (0,l):
            
            dot_product=0
            len_vec1=0
            len_vec2=0
            for k in range(len(word_tokens_refined)):
                dot_product+=Tf_Idf[i][k]*Tf_Idf[j][k]
                len_vec1+=Tf_Idf[i][k]*Tf_Idf[i][k]
                len_vec2+=Tf_Idf[j][k]*Tf_Idf[j][k]

            val=1
            if(len_vec1!=0 and len_vec2!=0):
                val=dot_product/(math.sqrt(len_vec1)*math.sqrt(len_vec2))
            cosine_value[sentence]+=val
    
    for i in range(0,l):
        sentence=sent_tokens[i]
        cosine_value[sentence]=cosine_value[sentence]/l
        
    cosine_value=sorted(cosine_value.items(), key=lambda x: x[1], reverse=True)
    
    #centrality
    cos_value={}
    factor=1/l;
    for i in range(len(cosine_value)):
        k=cosine_value[i][0]
        cos_value[k]=1-i*factor
        cos_value[k]=round(cos_value[k],3)
        i+=1
    
    for i in range(0,l):
        sentence=sent_tokens[i]
        centrality_value[sentence]=cos_value[sentence]
    
    print(centrality_value.values())
    return centrality_value


#..................... part 10(thematic).........................
def thematicFeature(sent_tokens) :
    word_list = []
    for sentence in sent_tokens :
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                #print(word)
                word_list.append(word)
            except Exception as e:
                print("ERR")
    counts = Counter(word_list)
    number_of_words = len(counts)
    most_common = counts.most_common(10)
    thematic_words = []
    for data in most_common :
        thematic_words.append(data[0])
    scores = []
    for sentence in sent_tokens :
        score = 0
        for word in sentence :
            try:
                word = ''.join(e for e in word if e.isalnum())
                if word in thematic_words :
                    score = score + 1
                #print(word)
            except Exception as e:
                print("ERR")
        score = 1.0*score/(number_of_words)
        scores.append(score)
    max_value=max(scores)
    if(max_value!=0):
        for k in range(len(scores)):
            scores[k] = (scores[k]/max_value)
            scores[k] = round(scores[k], 3)
    print(scores)
    return scores

#..................... part 11(Named Entity Recoginition).........................
def namedEntityRecog(sent_tokens) :
    counts = []
    for sentence in sent_tokens :
        count = ner(sentence)
        counts.append(count)
    max_value=max(counts)
    if(max_value!=0):
        for k in range(len(counts)):
            counts[k] = (counts[k]/max_value)
            counts[k] = round(counts[k], 3)
    print(counts)
    return counts


#..................... part 12(Pos tagging).........................
def posTagger(tokenized_sentences) :
    tagged = []
    for sentence in tokenized_sentences :
        tag = nltk.pos_tag(sentence)
        tagged.append(tag)
    print(tagged)
    return tagged

#..................... part 13(jaccards similarity).........................
def similarityScores(tokenized_sentences) :
    scores = []
    for sentence in tokenized_sentences :
        score = 0;
        for sen in tokenized_sentences :
            if sen != sentence :
                score += similar(sentence,sen)
        scores.append(score)
    max_value=max(scores)
    if(max_value!=0):
        for k in range(len(scores)):
            scores[k] = (scores[k]/max_value)
            scores[k] = round(scores[k], 3)
    print(scores)
    return scores


corpus=[]

def get_data(text,text1,flag):
    #sent_tokens = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])|\.(?=[^0-9])|\n', text)
    #sent_tokens = nltk.sent_tokenize(text)
    sent_tokens=split_into_sentences(text)
    word_tokens = nltk.word_tokenize(text)

    sent_tokens_temp=[]
    for sentence in sent_tokens:
        if(sentence==''):
            continue
        if(sentence in sent_tokens_temp):
            continue
        sent_tokens_temp.append(sentence)
    sent_tokens=sent_tokens_temp
    
    word_tokens_lower=[word.lower() for word in word_tokens]
    stopWords = list(set(stopwords.words("english")))
    word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]

    for sentence in sent_tokens:
        word_tokens=nltk.word_tokenize(sentence)
        word_tokens_lower=[word.lower() for word in word_tokens]
        stopWords = list(set(stopwords.words("english")))
        word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]
        word_tokens_refined = ' '.join(word_tokens_refined)
        corpus.append(word_tokens_refined)

    cv = TfidfVectorizer()
    g=cue_phrases(sent_tokens)
    z=list(g.keys())
    g=list(g.values())
    h=numeric_data(sent_tokens)
    h=list(h.values())
    i=sent_len_score(sent_tokens)
    i=list(i.values())
    j=sentence_position(sent_tokens)
    j=list(j.values())   
    p=upper_case(sent_tokens)
    p=list(p.values())
    l=head_match(sent_tokens)
    l=list(l.values())
    m=word_frequency(sent_tokens,word_tokens_refined)
    m=list(m.values())
    n=proper_noun(sent_tokens)
    n=list(n.values())
    c=centrality(sent_tokens,cv,word_tokens_refined)
    c=list(c.values())
    d=thematicFeature(sent_tokens)
    e=namedEntityRecog(sent_tokens)
    #q=posTagger(sent_tokens)
    r=similarityScores(sent_tokens)

    if(flag==0):
        total_score=[]
        sumValues=0
        for k in range(len(sent_tokens)):
            score=g[k]+h[k]+i[k]+j[k]+p[k]+l[k]+m[k]+n[k]+c[k]+d[k]+e[k]+r[k]
            total_score.append(score)
            sumValues+=score
        print(total_score)

        average = sumValues / len(total_score)
        print(average)

        # Storing sentences into our summary. 
        summary = '' 
        k=0
        for sentence in sent_tokens: 
            if (total_score[k] > (1.2*average)): 
                summary += " " + sentence 
            k+=1
        print(summary) 
        return summary

    #sent_tokens1 = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])|\.(?=[^0-9])|\n', text1)
    #sent_tokens1 = nltk.sent_tokenize(text1)
    sent_tokens1=split_into_sentences(text1)
    word_tokens1 = nltk.word_tokenize(text1)

    sent_tokens1_temp=[]
    for sentence in sent_tokens1:
        if(sentence==''):
            continue
        if(sentence in sent_tokens1_temp):
            continue
        sent_tokens1_temp.append(sentence)
    sent_tokens1=sent_tokens1_temp

    label={}
    for sentence in sent_tokens:
        label[sentence]=0
    for sent in sent_tokens1:
        if sent in sent_tokens:
            label[sent]=1
                    
    o=list(label.values())

    df=pd.DataFrame(columns=['cue_phrase','numerical_data','sent_length','sent_position','word_freq','upper','proper_nouns','head_matching','centrality','thematic','ner','jaccard','key','label'])
    df = df.append(pd.DataFrame({'cue_phrase': g,
                                 'numerical_data': h,
                                 'sent_length': i,
                                 'sent_position': j,
                                 'upper': p,
                                 'head_matching': l,
                                 'word_freq': m,
                                 'proper_nouns': n,
                                 'centrality':c,
                                 'thematic':d,
                                 'ner':e,
                                 'jaccard':r,
                                 'key': z,
                                 'label': o}), ignore_index=True)

    df['label']=df['label'].astype(int)

    columns=['cue_phrase','numerical_data','sent_length','sent_position','word_freq','upper','proper_nouns','head_matching','centrality','thematic','ner','jaccard']
    training=df[columns]
    test=df.label
    print(training)
    print(test)
    print(df.key)

    X_train, X_test, y_train, y_test = train_test_split(training, test, test_size=0.3)

    clf2 = LogisticRegression()
    #Train the model using the training sets
    clf2.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf2.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    accuracy=metrics.accuracy_score(y_test, y_pred)*100

    print(accuracy)
    return accuracy

