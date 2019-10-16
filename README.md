# Analysis and Topic Modeling on Twitter Data


### Models used:
- Latent Dirichlet Allocation (LDA)
    - Soft Clustering
- K-Means Clustering
    - Hard Clustering


### Notes about the data
- 5113 total documents
    - Sentiment
        - 519 documents labeled as having a positive sentiment
        - 572 documents labeled as having a negative sentiment
        - 2333 documents labeled as having a neutral sentiment
        - 1689 documents labeled as having an irrelevant sentiment
    - Topics
        - 1142 documents labeled as having a topic of Apple
        - 1317 documents labeled as having a topic of Google
        - 1364 documents labeled as having a topic of Microsoft
        - 1290 documents labeled as having a topic of Twitter


### Notes about NLP process
- Focused on LDA using gensim
- Used coherence plot to determine optimal number of topics after trying with 4.
- created a custom file of spanish stopwords since many tweets were in spanish.
    - List base taken from here: https://github.com/Alir3z4/stop-words/blob/master/spanish.txt
