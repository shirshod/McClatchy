#!/usr/bin/env python
# coding: utf-8

# ## State of the Union (1947-2022): Sentiment Analysis and Topic Modeling
# 
# By: Shirsho Dasgupta (2022)

# #### General Notes: 
# 
# Some visualizations may not be compatible with viewing directly on Github. View it by copying the URL [here](https://nbviewer.org/) or run the code after downloading this repo.
# 
# 
# #### Notes on Sentiment Analysis: 
# 
# The code reads the text of every State of the Union speech (delivered in-person on the Hill) from 1947 to 2022 and performs sentiment analysis on them using the NRC Word-Emotion Association Lexicon (EmoLex). 
# 
# The NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done by crowdsourcing.
# 
# The code matches the words in the speeches to that of the dictionary then adds up the factor of the emotion. 
# 
# Since EmoLex is, at the end of the day, crowdsourced and finite, there might be some words which are outside its scope or some nuances which are not accounted for (the code performs word-to-word comparison). The analysis is always at best an approximation. 
# 
# #### Notes on Topic Modeling: 
# 
# The code deploys machine-learning modules on the text of every State of the Union speech (delivered in-person on the Hill) from 1947 to 2022. The algorithm calculates the frequency of each word (controlling for how common they are etc.) and generates topics that were touched on. 
# 
# The code uses the Scikitlearn module to compute Non-Negative Matrix Factorization (NMF)/Latent Semantic Indexing (LSI) and Latent Dirichlet Allocation (LDA) methods. It also deploys Gensim to compute the same. 
# 
# #### Sources:
# 
# State of the Union [Archived Speeches](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union) at University of California, Santa Barbara.
# 
# [NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)

# ### Importing libraries

# In[1]:


import numpy as np
import re
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Creating dataframe of speeches

# #### Importing SOTU speeches

# In[2]:


import glob

filenames = glob.glob("sotu/*")


# In[3]:


speeches = [open(filename).read() for filename in filenames]
len(speeches)


# #### Storing SOTU speeches in one dataframe

# In[4]:


### stores entire text of speech under one column and the associated filename under another
speeches_df = pd.DataFrame({"text": speeches, "filename": filenames})
speeches_df.head(3)


# In[5]:


### creates two new columns to store year and name of president
speeches_df["year"] = " "
speeches_df["pres"] = " "
speeches_df["id"] = " "

### loop runs through dataframe
for i in range(0, len(speeches_df)):
    
    ### extracts (from filename) and stores year of speech
    speeches_df["year"][i] = int(speeches_df["filename"][i][-8:-4])
    x = speeches_df["year"][i]
    
    ### condition checks what year the speech was delivered and stores the name of the president accordingly
    if (x >= 1947) & (x <= 1952):
        speeches_df["pres"][i] = "truman"
    elif (x >= 1953) & (x <= 1960):
        speeches_df["pres"][i] = "eisenhower"
    elif (x >= 1961) & (x <= 1963):
        speeches_df["pres"][i] = "kennedy"
    elif (x >= 1964) & (x <= 1969):
        speeches_df["pres"][i] = "johnson"
    elif (x >= 1970) & (x <= 1974):
        speeches_df["pres"][i] = "nixon"
    elif (x >= 1975) & (x <= 1977):
        speeches_df["pres"][i] = "ford"
    elif (x >= 1978) & (x <= 1980):
        speeches_df["pres"][i] = "carter"
    elif (x >= 1981) & (x <= 1988):
        speeches_df["pres"][i] = "reagan"
    elif (x >= 1989) & (x <= 1992):
        speeches_df["pres"][i] = "bush sr."
    elif (x >= 1993) & (x <= 2000):
        speeches_df["pres"][i] = "clinton"
    elif (x >= 2001) & (x <= 2008):
        speeches_df["pres"][i] = "bush jr."
    elif (x >= 2009) & (x <= 2016):
        speeches_df["pres"][i] = "obama"
    elif (x >= 2017) & (x <= 2020):
        speeches_df["pres"][i] = "trump"
    elif (x > 2020):
        speeches_df["pres"][i] = "biden"
    
    speeches_df["id"][i] = speeches_df["pres"][i] + "-" + str(speeches_df["year"][i])


# In[6]:


### displays final master dataframe
speeches_df


# #### Creating separate dataframes for each president

# In[7]:


### filters by name of president and stores in separate dataframes
speeches_df_obama = speeches_df[speeches_df["pres"] == "obama"] 
speeches_df_trump = speeches_df[speeches_df["pres"] == "trump"] 
speeches_df_biden = speeches_df[speeches_df["pres"] == "biden"] 


# ### Sentiment analysis

# #### Importing NRC Emotion Lexicon

# Download compressed file from http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
# 
# Move the relevant text-file to the directory

# In[8]:


filepath = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emolex_df = pd.read_csv(filepath,  names = ["word", "emotion", "association"], sep = "\t", keep_default_na = False)
emolex_df = emolex_df.pivot(index = "word", columns = "emotion", values = "association").reset_index()
emolex_df.head()


# #### Calculating share of words using TfidfVectorizer

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(vocabulary = emolex_df.word, use_idf = False, norm = "l1") 
matrix = vec.fit_transform(speeches_df.text)
vocab = vec.get_feature_names()
wordcount_df = pd.DataFrame(matrix.toarray(), columns = vocab)
wordcount_df.head()


# #### Applying EmoLex

# The words which appear per category in the NRC Emotion Lexicon dictionary are identified and their shares are added up. The sum gives the total share of the document that relates to a particular emotion or sentiment. For instance say a document has two "angry" words each making up 0.01 of the text, then the anger-factor as it were of the text is 0.02 (= 0.01 + 0.01). 

# In[10]:


### identifies all negative words
neg_words = emolex_df[emolex_df.negative == 1]["word"]
### adds up shares of negative words
speeches_df["negative"] = wordcount_df[neg_words].sum(axis = 1)

### the above process is repeated per sentiment

pos_words = emolex_df[emolex_df.positive == 1]["word"]
speeches_df["positive"] = wordcount_df[pos_words].sum(axis = 1)

angry_words = emolex_df[emolex_df.anger == 1]["word"]
speeches_df["anger"] = wordcount_df[angry_words].sum(axis = 1)

anticip_words = emolex_df[emolex_df.anticipation == 1]["word"]
speeches_df["anticipation"] = wordcount_df[anticip_words].sum(axis = 1)

disgust_words = emolex_df[emolex_df.disgust == 1]["word"]
speeches_df["disgust"] = wordcount_df[disgust_words].sum(axis = 1)

fear_words = emolex_df[emolex_df.fear == 1]["word"]
speeches_df["fear"] = wordcount_df[fear_words].sum(axis = 1)

joy_words = emolex_df[emolex_df.joy == 1]["word"]
speeches_df["joy"] = wordcount_df[joy_words].sum(axis = 1)

sad_words = emolex_df[emolex_df.sadness == 1]["word"]
speeches_df["sadness"] = wordcount_df[sad_words].sum(axis = 1)

surprise_words = emolex_df[emolex_df.surprise == 1]["word"]
speeches_df["surprise"] = wordcount_df[surprise_words].sum(axis = 1)

trust_words = emolex_df[emolex_df.trust == 1]["word"]
speeches_df["trust"] = wordcount_df[trust_words].sum(axis = 1)


# In[11]:


### displays final dataframe
speeches_df.head()


# #### Exporting the dataset

# In[12]:


speeches_df.to_csv("sotu_analysis.csv", index = False)


# #### Re-importing created dataset 

# In[13]:


speeches = pd.read_csv("sotu_analysis.csv")


# In[14]:


speeches.shape


# #### Plotting sentiments

# ##### Importing Altair libraries

# In[15]:


import altair as alt

alt.renderers.enable("default")


# ##### Plotting anger-factor by president

# In[16]:


alt.Chart(speeches).mark_circle(size = 100).encode(
    x = "pres",
    y = "anger",
    tooltip = ["pres", "year", "negative"]
).configure_axis(
    grid = True
).properties(
    width = 650,
    height = 350,title = {
      "text": "Anger vs. president", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).configure_axisX(
    labelAngle = 45
).interactive()


# ##### Plotting negativity-factor by year

# In[17]:


### creates chart with point-markings
points = alt.Chart(speeches).mark_circle(size = 100).encode(
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    alt.Y("negative"),
    alt.Color("pres"),
    tooltip = ["pres", "year", "negative"] ### adds tooltip on mouse-hover
)

### creates line chart
line = alt.Chart(speeches).mark_line().encode(
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    alt.Y("negative"),
)

### layers both charts and adds some other properties
(line + points).configure_axis(
    grid = True
).configure_axisX(
    labelAngle = 45
).properties(
    width = 810,
    height = 250,
    title = {
      "text": "Negativity vs. year", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).interactive()


# ### Topic modeling

# #### Setting dataframe

# In[18]:


sotu = speeches[["text", "filename", "year", "pres", "id"]].copy()
sotu.head()


# #### Non-Negative Matrix Factorization (NMF)/Latent Semantic Indexing (LSI) topic modeling

# ##### Vectorizing all words

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

class StemmedTfidVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidVectorizer, self).build_analyzer()
        return lambda doc:(stemmer.stem(word) for word in analyzer(doc))

tfidf_vectorizer = StemmedTfidVectorizer(stop_words = "english", min_df = 5, max_df = 0.40)
x = tfidf_vectorizer.fit_transform(sotu.text.astype(str))


# ##### Deploying NMF module

# In[20]:


get_ipython().run_cell_magic('time', '', '\n### imports modules\nfrom sklearn.decomposition import NMF\n\n### selects number of topics \nmodel = NMF(n_components = 10)\nmodel.fit(x)\n\n### selects number of words per model\nn_words = 7\n### stores each word\nfeature_names = tfidf_vectorizer.get_feature_names()\n\n### creates new arrays to store list of topics and topic numbers\ntopic_list_nmf = []\ntopic_list_index_nmf = []\n\n### applies NMF modeling\nfor topic_idx, topic in enumerate(model.components_):\n    top_n = [feature_names[i]\n             for i in topic.argsort()\n             [-n_words:]][::-1]\n    top_features = \' \'.join(top_n)\n    topic_list_nmf.append(f"{\'_\'.join(top_n[:3])}") \n    topic_list_index_nmf.append("Topic " + str(topic_idx + 1))\n    \n    ### displays topics\n    print("Topic " + str(topic_idx + 1) + ": " + top_features)')


# ##### Converting computed factors into dataset

# In[21]:


### converts counts into numbers
amounts = model.transform(x) * 100

### sets up dataframe with corresponding topic-numbers and amount numbers
topics_nmf = pd.DataFrame(amounts, columns = topic_list_index_nmf)

### sets up dataframe with corresponding topics and subjects
topics_list_nmf = pd.DataFrame(amounts, columns = topic_list_nmf)

topics_nmf


# ##### Storing topic list as a dataframe

# In[22]:


topic_indices = list(topics_nmf)
topic_subjects = list(topics_list_nmf)
topic_desc = pd.DataFrame(topic_indices, topic_subjects).reset_index()
topic_desc_nmf = topic_desc.rename(columns = {"index": "topic_desc", 0: "topic"})
topic_desc_nmf


# ##### Merging speeches database with topics dataframe

# In[23]:


nmf_merged = topics_nmf.join(sotu)
nmf_merged.head(3)


# ##### Rearranging dataframe for plotting

# In[24]:


nmf_1 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 1"]].copy()
nmf_1["topic"] = "Topic 1"
nmf_1 = nmf_1.rename(columns = {"Topic 1": "factor"})

nmf_2 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 2"]].copy()
nmf_2["topic"] = "Topic 2"
nmf_2 = nmf_2.rename(columns = {"Topic 2": "factor"})

nmf_3 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 3"]].copy()
nmf_3["topic"] = "Topic 3"
nmf_3 = nmf_3.rename(columns = {"Topic 3": "factor"})

nmf_4 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 4"]].copy()
nmf_4["topic"] = "Topic 4"
nmf_4 = nmf_4.rename(columns = {"Topic 4": "factor"})

nmf_5 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 5"]].copy()
nmf_5["topic"] = "Topic 5"
nmf_5 = nmf_5.rename(columns = {"Topic 5": "factor"})

nmf_6 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 6"]].copy()
nmf_6["topic"] = "Topic 6"
nmf_6 = nmf_6.rename(columns = {"Topic 6": "factor"})

nmf_7 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 7"]].copy()
nmf_7["topic"] = "Topic 7"
nmf_7 = nmf_7.rename(columns = {"Topic 7": "factor"})

nmf_8 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 8"]].copy()
nmf_8["topic"] = "Topic 8"
nmf_8 = nmf_8.rename(columns = {"Topic 8": "factor"})

nmf_9 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 9"]].copy()
nmf_9["topic"] = "Topic 9"
nmf_9 = nmf_9.rename(columns = {"Topic 9": "factor"})

nmf_10 = nmf_merged[["text", "filename", "year", "pres", "id", "Topic 10"]].copy()
nmf_10["topic"] = "Topic 10"
nmf_10 = nmf_10.rename(columns = {"Topic 10": "factor"})

nmf_master = pd.concat([nmf_1, nmf_2, nmf_3, nmf_4, nmf_5, nmf_6, nmf_7, nmf_8, nmf_9, nmf_10],ignore_index = True)
nmf_master = pd.merge(nmf_master, topic_desc_nmf, on = "topic", how = "left")

nmf_master


# ##### Plotting steamgraph

# In[25]:


alt.Chart(nmf_master).mark_area(opacity = 0.5).encode(
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    alt.Y("sum(factor):Q", stack = "center", axis = None),
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    tooltip = ["pres", "year", "topic_desc"]
).configure_axis(
    grid = True
).configure_axisX(
    labelAngle = 45
).properties(
    width = 750,
    height = 250,
    title = {
      "text": "Topics by year", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).interactive()


# ##### Plotting line chart

# In[26]:


lines = alt.Chart(nmf_master).mark_line().encode(
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    y = "factor",
)

points = alt.Chart(nmf_master).mark_point().encode(
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    y = "factor",
    tooltip = ["pres", "year", "topic_desc"]   
)

chart = lines + points

chart.configure_axis(
    grid = True
).configure_axisX(
    labelAngle = 45
).properties(
    width = 750,
    height = 250,
    title = {
      "text": "Topics by year", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).interactive()


# Note: This same figure can also be drawn using selected topics.

# #### Latent Dirichlet Allocation (LDA) topic modeling

# ##### Displaying dataframe

# In[27]:


sotu


# ##### Vectorizing all words

# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

analyzer = CountVectorizer().build_analyzer()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))

vectorizer = StemmedCountVectorizer(stop_words = "english", min_df = 5, max_df = 0.4)
x = vectorizer.fit_transform(sotu.text)


# Note: LDA modeling has in-built TF-IDF unlike NMF, so only a CountVectorizer is required.

# ##### Deploying LDA module

# In[29]:


get_ipython().run_cell_magic('time', '', '\n### imports modules\nfrom sklearn.decomposition import LatentDirichletAllocation\n\n### assigns number of topics\nn_topics = 10\nmodel = LatentDirichletAllocation(n_components = n_topics)\nmodel.fit(x)\n\n### assigns number of words per topic\nn_words = 7\nfeature_names = vectorizer.get_feature_names()\n\n### applies LDA modeling\ntopic_list_lda = []\ntopic_list_index_lda = []\nfor topic_idx, topic in enumerate(model.components_):\n    top_n = [feature_names[i]\n             for i in topic.argsort()\n             [-n_words:]][::-1]\n    top_features = " ".join(top_n)\n    topic_list_lda.append(f"{\'_\'.join(top_n[:3])}") \n    topic_list_index_lda.append("Topic " + str(topic_idx + 1))\n\n    print(f"Topic {topic_idx + 1}: {top_features}")')


# ##### Converting computed factors into dataset

# In[30]:


### converts counts into numbers
amounts = model.transform(x) * 100

### sets up dataframe with corresponding topic-numbers and amount numbers
topics_lda = pd.DataFrame(amounts, columns = topic_list_index_lda)

### sets up dataframe with corresponding topics and subjects
topics_list_lda = pd.DataFrame(amounts, columns = topic_list_lda)

topics_lda


# ##### Storing topics list as a dataframe

# In[31]:


topic_indices = list(topics_lda)
topic_subjects = list(topics_list_lda)
topic_desc = pd.DataFrame(topic_indices, topic_subjects).reset_index()
topic_desc_lda = topic_desc.rename(columns = {"index": "topic_desc", 0: "topic"})
topic_desc_lda


# ##### Merging speeches database with topics dataframe

# In[32]:


lda_merged = topics_lda.join(sotu)
lda_merged.head(3)


# ##### Rearranging dataframe for plotting

# In[33]:


lda_1 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 1"]].copy()
lda_1["topic"] = "Topic 1"
lda_1 = lda_1.rename(columns = {"Topic 1": "factor"})

lda_2 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 2"]].copy()
lda_2["topic"] = "Topic 2"
lda_2 = lda_2.rename(columns = {"Topic 2": "factor"})

lda_3 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 3"]].copy()
lda_3["topic"] = "Topic 3"
lda_3 = lda_3.rename(columns = {"Topic 3": "factor"})

lda_4 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 4"]].copy()
lda_4["topic"] = "Topic 4"
lda_4 = lda_4.rename(columns = {"Topic 4": "factor"})

lda_5 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 5"]].copy()
lda_5["topic"] = "Topic 5"
lda_5 = lda_5.rename(columns = {"Topic 5": "factor"})

lda_6 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 6"]].copy()
lda_6["topic"] = "Topic 6"
lda_6 = lda_6.rename(columns = {"Topic 6": "factor"})

lda_7 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 7"]].copy()
lda_7["topic"] = "Topic 7"
lda_7 = lda_7.rename(columns = {"Topic 7": "factor"})

lda_8 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 8"]].copy()
lda_8["topic"] = "Topic 8"
lda_8 = lda_8.rename(columns = {"Topic 8": "factor"})

lda_9 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 9"]].copy()
lda_9["topic"] = "Topic 9"
lda_9 = lda_9.rename(columns = {"Topic 9": "factor"})

lda_10 = lda_merged[["text", "filename", "year", "pres", "id", "Topic 10"]].copy()
lda_10["topic"] = "Topic 10"
lda_10 = lda_10.rename(columns = {"Topic 10": "factor"})

lda_master = pd.concat([lda_1, lda_2, lda_3, lda_4, lda_5, lda_6, lda_7, lda_8, lda_9, lda_10], ignore_index = True)
lda_master = pd.merge(lda_master, topic_desc_lda, on = "topic", how = "left")

lda_master


# ##### Plotting steamgraph

# In[34]:


alt.Chart(lda_master).mark_area(opacity = 0.5).encode(
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    alt.Y("sum(factor):Q", stack = "center", axis = None),
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    tooltip = ["pres", "year", "topic_desc"]
).configure_axis(
    grid = True
).configure_axisX(
    labelAngle = 45
).properties(
    width = 750,
    height = 250,
    title = {
      "text": "Topics by year", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).interactive()


# ##### Plotting line chart

# In[35]:


lines = alt.Chart(lda_master).mark_line().encode(
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    y = "factor",
)

points = alt.Chart(lda_master).mark_point().encode(
    alt.Color("topic_desc", scale = alt.Scale(scheme = "tableau10")),
    alt.X("year:O", axis = alt.Axis(values = list(range(1947, 2022, 5)))),
    y = "factor",
    tooltip = ["pres", "year", "topic_desc"]   
)

chart = lines + points

chart.configure_axis(
    grid = True
).configure_axisX(
    labelAngle = 45
).properties(
    width = 750,
    height = 250,
    title = {
      "text": "Topics by year", 
      "subtitle": "Hover over the figure for tooltip",
    }
).configure_title(
    fontSize = 15,
    anchor = "start",
).interactive()


# #### Comparing NMF and LDA models:
# 
# The topics generated by both models are nearly the identical but the ones from NMF seem to make a bit more sense. 
# 
# Note: When applying topic modeling, sometimes which method is chosen boils down to a subjective decision. GridSearchCV can be used to deploy an algorithm to suggest the best parameters but that too is not always foolproof. The primary question should always be "What makes sense?"

# #### Gensim modeling

# ##### Setting dataframe

# In[36]:


sotu.text = sotu.text.str.replace("[^A-Za-z ]", " ")

sotu.head(3)


# ##### Deploying LSI/NMF modeling with Gensim

# In[37]:


### imports modules
from gensim.utils import simple_preprocess

### stores text of speeches and pre-processes 
texts = sotu.text.apply(simple_preprocess)

### imports modules
from gensim import corpora

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below = 5, no_above = 0.4)
corpus = [dictionary.doc2bow(text) for text in texts]

### imports modules
from gensim import models

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

### assigns number of topics
n_topics = 10

### builds a model
nmf_model = models.LsiModel(corpus_tfidf,
                            id2word = dictionary,
                            num_topics = n_topics)

### displays topic results
nmf_model.print_topics()


# In[38]:


### assigns number of words per topic
n_words = 10

topic_words = pd.DataFrame({})

### computes dataframe with results
for i, topic in enumerate(nmf_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({"value": feature_values, "word": words, "topic": (i + 1)})
    topic_words = pd.concat([topic_words, topic_df], ignore_index = True)

topic_words.head()


# ##### Visualizing Gensim-LSI/NMF model using Seaborn

# In[39]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

g = sns.FacetGrid(topic_words, col = "topic", col_wrap = 5, sharey = False)
g.map(plt.barh, "word", "value")


# ##### Deploying LDA model with Gensim

# In[40]:


### imports modules
from gensim.utils import simple_preprocess

### stores text of speeches and pre-processes
texts = sotu.text.apply(simple_preprocess)

### imports modules
from gensim import corpora

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below = 5, no_above = 0.4, keep_n = 2000)
corpus = [dictionary.doc2bow(text) for text in texts]

### assigns number of topics
from gensim import models

### builds a model
n_topics = 10
lda_model = models.LdaModel(corpus = corpus, num_topics = n_topics)

### displays topic results
lda_model.print_topics()


# In[41]:


### assigns number of words per topic
n_words = 10

topic_words = pd.DataFrame({})

### computes dataframe with results
for i, topic in enumerate(lda_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({"value": feature_values, "word": words, "topic": (i + 1)})
    topic_words = pd.concat([topic_words, topic_df], ignore_index = True)

topic_words.head()


# ##### Visualizing Gensim-LDA model using Seaborn

# In[42]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns

g = sns.FacetGrid(topic_words, col = "topic", col_wrap = 5, sharey = False)
g.map(plt.barh, "word", "value")


# ##### Visualizing Gensim-LDA model using PyLDAvis

# In[43]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, dictionary)
vis


# In[ ]:




