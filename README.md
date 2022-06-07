# State of the Union (1947-2022): Sentiment Analysis and Topic Modeling

By: Shirsho Dasgupta (2022)

#### General Notes: 

The Python notebook is too large for Github or NB Viewer to render and moreover contains interactive Altair charts which Github does not render. 
Download the repo and run it locally on your computer. 
A view of the notebook in PDF format can be [found here](https://github.com/shirshod/mcclatchy/blob/main/sotu_analysis/sotu_analysis.pdf).
The Python code in .py format can be [found here](https://github.com/shirshod/mcclatchy/blob/main/sotu_analysis/sotu_analysis.py).

#### Notes on Sentiment Analysis: 

The code reads the text of every State of the Union speech (delivered in-person on the Hill) from 1947 to 2022 and performs sentiment analysis on them using the NRC Word-Emotion Association Lexicon (EmoLex). 

The NRC Emotion Lexicon is a list of English words and their associations with eight basic emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive). The annotations were manually done by crowdsourcing.

The code matches the words in the speeches to that of the dictionary then adds up the factor of the emotion. 

Since EmoLex is, at the end of the day, crowdsourced and finite, there might be some words which are outside its scope or some nuances which are not accounted for (the code performs word-to-word comparison). The analysis is always at best an approximation. 

#### Notes on Topic Modeling: 

The code deploys machine-learning modules on the text of every State of the Union speech (delivered in-person on the Hill) from 1947 to 2022. The algorithm calculates the frequency of each word (controlling for how common they are etc.) and generates topics that were touched on. 

The code uses the Scikitlearn module to compute Non-Negative Matrix Factorization (NMF)/Latent Semantic Indexing (LSI) and Latent Dirichlet Allocation (LDA) methods. It also deploys Gensim to compute the same. 

#### Sources:

State of the Union [Archived Speeches](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union) at University of California, Santa Barbara.

[NRC Emotion Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
