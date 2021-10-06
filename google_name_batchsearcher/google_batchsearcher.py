#!/usr/bin/env python
# coding: utf-8

# ### Google Names Batch Search
# 
# By: Shirsho Dasgupta (2021) 

# The Miami Herald often works on investigations based on corporate records — sometimes public, at other times leaked. These records often have — or reporters can make it themselves — lists of companies, their owners and/or directors and other officers. 
# 
# This project was initiated to automate an initial search on who these people are. 
# 
# The code imports a spreadsheet with a list of names then searches for them in Google. It then extracts the first few lines about that person that come up as flashcard in a regular Google search.  
# 
# An short example is attached. 
# 
# The file that is imported is names.csv
# 
# The resulting file is search_results.csv
# 
# ##### Notes:
# 
# 1. This search is only to be used as a starting point. The results are not fully confirmed. Some of the ways in which one can obtain a complete confirmation is to match DOBs or photos. 
# 
# 2. Overloading Google with queries can make their networks label the code as a bot and block access. Care must be taken to break the searches up and have sleep times between each iteration.

# ### Importing libraries

# In[1]:


import requests
import bs4
import pandas as pd
import time


# ### Importing spreadsheet for batch of names to be searched

# In[2]:


searchlist = pd.read_csv("names.csv")  
searchlist.head(5)


# ### Preparing dataframe and running search

# In[3]:


## adding columns to be filled in from google
searchlist["Googled_Names"] = " "
searchlist["Descriptor_1"] = " "
searchlist["Descriptor_2"] = " "
searchlist["Descriptor_3"] = " "
searchlist["Descriptor_4"] = " "
searchlist["Descriptor_5"] = " "


# In[4]:


## storing number of rows in the spreadsheet
rows = searchlist.shape[0] 

## setting up loop to run through each row
for i in range(0, rows):
    
    ## concatenating with "+" sign if a cell has multiple words for google search url pattern
    txt = searchlist["Names"][i]
    terms = "+"
    x = txt.split()
    terms = terms.join(x)
    
    ## storing url
    url = "https://google.com/search?q=" + terms
    
    ## getting url and converting for scrape
    request_result = requests.get(url)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")
    
    ## setting up exception handling, if there is a result the search details are stored, if not, loops moves onto next row
    try:
        
        ## finds "div" tag and the class that stores the names and descriptors; note: this sometimes changes and should be checked and modified accordingly
        heading_object = soup.find_all("div", class_= "BNeawe")
        
        ## runs through each of the entries; relevant information is generally stored in the first six cells
        for info in heading_object:
            names = heading_object
        
        ## writes results into the relevant results column
        searchlist["Googled_Names"][i] = names[0].text
        searchlist["Descriptor_1"][i] = names[1].text
        searchlist["Descriptor_2"][i] = names[2].text
        searchlist["Descriptor_3"][i] = names[3].text
        searchlist["Descriptor_4"][i] = names[5].text
        searchlist["Descriptor_5"][i] = names[6].text
    except:
        i = i + 1
        
    ## sleeper ensures that google does not mistake script for a bot and blocks access    
    time.sleep(0.2)   


# In[5]:


## displaying results
searchlist.head(5)


# ### Exporting spreadsheet

# In[6]:


searchlist.to_csv("search_results.csv", index = False)

