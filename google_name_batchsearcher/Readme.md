### Google Names Batch Search

By: Shirsho Dasgupta (2021) 

The Miami Herald often works on investigations based on corporate records — sometimes public, at other times leaked. These records often have — or reporters can make it themselves — lists of companies, their owners and/or directors and other officers. 

This project was initiated to automate an initial search on who these people are. 

The code imports a spreadsheet with a list of names then searches for them in Google. It then extracts the first few lines about that person that come up as flashcard in a regular Google search.  

A short example is attached. 

The file that is imported is names.csv

The resulting file is search_results.csv

##### Notes:

1. This search is only to be used as a starting point. The results are not fully confirmed. Some of the ways in which one can obtain a complete confirmation is to match DOBs or photos. 

2. Overloading Google with queries can make their networks label the code as a bot and block access. Care must be taken to break the searches up and have sleep times between each iteration.

### Importing libraries


```python
import requests
import bs4
import pandas as pd
import time
```

### Importing spreadsheet for batch of names to be searched


```python
searchlist = pd.read_csv("names.csv")  
searchlist.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mark Zuckerberg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tony Blair</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joe Biden</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Steve Jobs</td>
    </tr>
  </tbody>
</table>
</div>



### Preparing dataframe and running search


```python
## adding columns to be filled in from google
searchlist["Googled_Names"] = " "
searchlist["Descriptor_1"] = " "
searchlist["Descriptor_2"] = " "
searchlist["Descriptor_3"] = " "
searchlist["Descriptor_4"] = " "
searchlist["Descriptor_5"] = " "
```


```python
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
```


```python
## displaying results
searchlist.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Names</th>
      <th>Googled_Names</th>
      <th>Descriptor_1</th>
      <th>Descriptor_2</th>
      <th>Descriptor_3</th>
      <th>Descriptor_4</th>
      <th>Descriptor_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump</td>
      <td>Donald Trump</td>
      <td>45th U.S. President · donaldjtrump.com</td>
      <td>Born: June 14, 1946 (age 75 years), Jamaica Ho...</td>
      <td>Height: 6′ 3″</td>
      <td>Party: Republican Party</td>
      <td>Spouse: Melania Trump (m. 2005), Marla Maples ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mark Zuckerberg</td>
      <td>Mark Zuckerberg</td>
      <td>Chief Executive Officer of Facebook</td>
      <td>View all</td>
      <td>Mark Elliot Zuckerberg is an American media ma...</td>
      <td>Net worth: 122.7 billion USD (2021)</td>
      <td>Born: May 14, 1984 (age 37 years), White Plain...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tony Blair</td>
      <td>Tony Blair</td>
      <td>Former Prime Minister of the United Kingdom</td>
      <td>Anthony Charles Lynton Blair is a British poli...</td>
      <td>Anthony Charles Lynton Blair is a British poli...</td>
      <td>Height: 6′ 0″</td>
      <td>Spouse: Cherie Blair (m. 1980)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joe Biden</td>
      <td>Joe Biden</td>
      <td>46th U.S. President · whitehouse.gov</td>
      <td>Joseph Robinette Biden Jr. is an American poli...</td>
      <td>Joseph Robinette Biden Jr. is an American poli...</td>
      <td>Born: November 20, 1942 (age 78 years), Scrant...</td>
      <td>Height: 6′ 0″</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Steve Jobs</td>
      <td>Steve Jobs</td>
      <td>American business magnate</td>
      <td>View all</td>
      <td>Steven Paul Jobs was an American business magn...</td>
      <td>Born: February 24, 1955, San Francisco, CA</td>
      <td>Died: October 5, 2011, Palo Alto, CA</td>
    </tr>
  </tbody>
</table>
</div>



### Exporting spreadsheet


```python
searchlist.to_csv("search_results.csv", index = False)
```
