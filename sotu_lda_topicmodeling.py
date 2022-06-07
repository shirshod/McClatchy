#!/usr/bin/env python
# coding: utf-8

# ## State of the Union (1947-2022): LDA Topic Modeling
# 
# By: Shirsho Dasgupta (2022)

# #### Notes: 
# 
# Some visualizations may not be compatible with viewing directly on Github. View it by copying the URL [here](https://nbviewer.org/) or run the code after downloading this repo.
# 
# The code deploys machine-learning modules on the text of every State of the Union speech (delivered in-person on the Hill) from 1947 to 2022. The algorithm calculates the frequency of each word (controlling for how common they are etc.) and generates topics that were touched on. 
# 
# The code uses the Scikitlearn module to compute Latent Dirichlet Allocation (LDA) methods. 
# 
# #### Sources:
# 
# State of the Union [Archived Speeches](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union) at University of California, Santa Barbara.

# ### Importing libraries

# In[1]:


import numpy as np
import re
import pandas as pd


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


# ### Latent Dirichlet Allocation (LDA) topic modeling

# #### Setting dataframe

# In[7]:


sotu = speeches_df.copy()
sotu.head()


# #### Vectorizing all words

# In[8]:


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

# #### Deploying LDA module

# In[9]:


get_ipython().run_cell_magic('time', '', '\n### imports modules\nfrom sklearn.decomposition import LatentDirichletAllocation\n\n### assigns number of topics\nn_topics = 10\nmodel = LatentDirichletAllocation(n_components = n_topics)\nmodel.fit(x)\n\n### assigns number of words per topic\nn_words = 7\nfeature_names = vectorizer.get_feature_names()\n\n### applies LDA modeling\ntopic_list_lda = []\ntopic_list_index_lda = []\nfor topic_idx, topic in enumerate(model.components_):\n    top_n = [feature_names[i]\n             for i in topic.argsort()\n             [-n_words:]][::-1]\n    top_features = " ".join(top_n)\n    topic_list_lda.append(f"{\'_\'.join(top_n[:3])}") \n    topic_list_index_lda.append("Topic " + str(topic_idx + 1))\n\n    print(f"Topic {topic_idx + 1}: {top_features}")')


# #### Converting computed factors into dataset

# In[10]:


### converts counts into numbers
amounts = model.transform(x) * 100

### sets up dataframe with corresponding topic-numbers and amount numbers
topics_lda = pd.DataFrame(amounts, columns = topic_list_index_lda)

### sets up dataframe with corresponding topics and subjects
topics_list_lda = pd.DataFrame(amounts, columns = topic_list_lda)

topics_lda


# #### Storing topics list as a dataframe

# In[11]:


topic_indices = list(topics_lda)
topic_subjects = list(topics_list_lda)
topic_desc = pd.DataFrame(topic_indices, topic_subjects).reset_index()
topic_desc_lda = topic_desc.rename(columns = {"index": "topic_desc", 0: "topic"})
topic_desc_lda


# #### Merging speeches database with topics dataframe

# In[12]:


lda_merged = topics_lda.join(sotu)
lda_merged.head(3)


# #### Rearranging dataframe for plotting

# In[13]:


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


# #### Plotting steamgraph

# In[14]:


import altair as alt
alt.renderers.enable("default")

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


# #### Plotting line chart

# In[15]:


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

# In[ ]:




