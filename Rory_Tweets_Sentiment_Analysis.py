## Shriya Kunatharaju
## Intro to Python FINAL
## 12/13/2022

## Using NLP to Understand Rory McIlroy's Major Performance

import pandas as pd
import re
import numpy as np
import statistics
import nltk                                                                     #Natural Language ToolKit
from nltk.sentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
import matplotlib.pyplot as plt
import ssl                                                                      #Needed to download stopwords from NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

##
## 1. TRANSCRIPT DATA CLEANING
##

data = pd.read_csv("Test.csv")                                                 #Transcript data from ASAPSports

data['text2'] = data['text'].str.replace("Â", "")                              #Replacing special characters that were parsed weirdly
data['text2'] = data['text2'].str.replace("‑", " ")                            #Adding text to text2

names = []                                                                     #Finding names of interviewers using RegEx
for i in range(len(data)):                                                     #CAPITALLETTERS *space* CAPITALLETTERS *colon*
    unique = set(re.findall(r'[A-Z]* [A-Z]*:', data.iloc[i,0]))
    names.extend(unique)
names = set(names)

data['text2'] = data['text2'].str.replace("MIKE WOODCOCK:", "Q.")              #Interviewer names replaced with "Q."
data['text2'] = data['text2'].str.replace("THE MODERATOR:", "Q.")
data['text2'] = data['text2'].str.replace("JOHN DEVER:", "Q.")
data['text2'] = data['text2'].str.replace("JULIUS MASON:", "Q.")
data['text2'] = data['text2'].str.replace("KELLY ELBIN:", "Q.")
data['text2'] = data['text2'].str.replace("Q.", "delim Q:")
data['text2'] = data['text2'].str.replace("RORY MCILROY:", "delim RORY MCILROY: ")     #Adding "delim" to be used as a delimiter to separate Interviewer/ Response Sections
data['text2'] = data['text2'].str.replace("RORY McILROY:", "delim RORY MCILROY: ")

def text_function(text):
    split_text = re.split("(delim)", text)                                     #Splitting text at delim, keeping if starts with RORY
    rory_text = [x for x in split_text if x.startswith(" RORY MCILROY")]

    merged_text = ""

    for x in rory_text:
        clean_text = x.replace(" RORY MCILROY: ", "")
        clean_text = clean_text.replace("\n", "")                             #Removing 'new line'
        merged_text = merged_text + clean_text

    return merged_text

data['text3'] = data['text2'].apply(text_function)                            #Applying function on text2 to get text3

def rm_words_in_par(text):
    text = re.sub(r"[(][a-zA-Z0-9 ]+[)]", "", text)                           #Removing words in parentheses, "(laughter)", "(inaudible)", etc.
    return text

data['text3'] = data['text3'].apply(rm_words_in_par)
data['text3'] = data['text3'].str.replace("End of FastScripts", "")
data['text3'] = data['text3'].str.replace("FastScripts Transcript by ASAP Sports", "")   #Removing FastScripts signature

data.to_csv("Test_2.csv")                                                     #Wrote data and spot checked 10 transcripts

text_df = data[['date', 'tournament', 'round', 'text3']]                      #Keeping relevant columns, getting year from date
text_df['date'] = pd.to_datetime(text_df.date)
text_df['date'] = text_df['date'].dt.year

##
## 2. SENTIMENT ANALYSIS
##

nltk.download(["names","stopwords", "averaged_perceptron_tagger", "vader_lexicon","punkt"])  #downloading relevant modules
stopwords = nltk.corpus.stopwords.words("english")

#test = text_df['text3'][1]

#tokens = nltk.word_tokenize(test)                                     #used for exploratory analysis, kept most interesting scores below
#words = [w for w in tokens if w not in stopwords]
#words = [w for w in words if str(w).isalpha() is True]
#fd = nltk.FreqDist(words)
#text = nltk.Text(words)
#text.concordance("good", lines=5)
#finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
#finder.ngram_fd.most_common(3)

def sentiment_analysis_neg(text):
    text = text.lower()                                          #gets negative sentiment score

    sia = SentimentIntensityAnalyzer()
    sia = sia.polarity_scores(text)

    neg = sia['neg']
    return neg

def sentiment_analysis_pos(text):
    text = text.lower()                                          #gets positive sentiment score

    sia = SentimentIntensityAnalyzer()
    sia = sia.polarity_scores(text)

    pos = sia['pos']
    return pos

text_df['neg'] = text_df['text3'].apply(sentiment_analysis_neg)       #adding sentiment scores to text_df
text_df['pos'] = text_df['text3'].apply(sentiment_analysis_pos)

def score_by_sent(text):
    scores = []                                             #to see sentiment by sentence, compound score accounts for negative and positive
    for sentence in nltk.sent_tokenize(text):
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(sentence)["compound"]
        scores.append(score)

    mean = round(statistics.mean(scores), 5)
    return mean

#text_df['mean_comp'] = text_df['text3'].apply(score_by_sent)     #adding mean compound score to text_df

#text_df2 = text_df[['date', 'tournament', 'round', 'neg', 'pos', 'mean_comp']]     #making long data wide
#text_df2 = pd.pivot(text_df2, columns= 'round', values= ['neg', 'pos', 'mean_comp'], index = ['date', 'tournament']).reset_index()

#text_df2.to_csv("text_df2.csv")

text_df2 = pd.read_csv("text_df2.csv")      #writting dataset to csv and changing column names manually for ease of use

##
## 3. TOURNAMENT STATISTICS
##

stats_df = pd.read_csv("PGA_Stats.csv")                                   #PGA tournament stats from Golf Stats

stats_df['finish'] = stats_df['finish'].str.replace("CUT","NA")           #Changing CUT to NA
stats_df['finish'] = stats_df['finish'].str.replace("T","")               #Removing tied from ranking
stats_df['finish'] = stats_df['finish'].str.replace("Win","1")            #Changing win to 1
stats_df['to_par'] = stats_df['to_par'].str.replace("E","0")              #Changing par to 0

def text_to_int(val):
    val = str(val)                                                        #function to change string to integer or NA values
    if re.match("\d+.*\d*",val):
        val = np.int64(float(val))
    else:
        val = np.nan
    return val

cols = ['finish','Score_R1','Score_R2','Score_R3','Score_R4','Pos_R1','Pos_R2','Pos_R3','to_par']

for col in cols:
    stats_df[col] = stats_df[col].apply(text_to_int)                     #Applying integer function to relevant columns


##
## 4. MERGE DATASETS
##

merged_df = pd.merge(text_df2, stats_df, on = ['date', 'tournament'])    #Merged text and stats dataframes


##
## 5. STATISTICS
##

y = merged_df['finish']                                 #Multi Linear Regression to see if sentiment predicts finish
x = merged_df[['neg1', 'neg2', 'neg3', 'neg4']]
x = sm.add_constant(x)
model = sm.OLS(y, x,missing='drop').fit()               #Model for negative sentiment

y = merged_df['finish']                                 #Multi Linear Regression to see if score and sentiment behaved the same
x = merged_df[['Score_R1', 'Score_R2', 'Score_R3', 'Score_R4']]
x = sm.add_constant(x)
model = sm.OLS(y, x,missing='drop').fit()

y = merged_df['finish']                                 #Multi Linear Regression to see if score and sentiment behaved the same
x = merged_df[['Pos_R1', 'Pos_R2', 'Pos_R3', 'finish']]
x = sm.add_constant(x)
model = sm.OLS(y, x,missing='drop').fit()

#print(model.summary())

y = merged_df['finish']
x = merged_df[['pos1', 'pos2', 'pos3', 'pos4']]
x = sm.add_constant(x)
model = sm.OLS(y, x,missing='drop').fit()               #Model for positive sentiment

y = merged_df['finish']
x = merged_df[['mean_comp1', 'mean_comp2', 'mean_comp3', 'mean_comp4']]
x = sm.add_constant(x)
model = sm.OLS(y, x,missing='drop').fit()               #Model for compound sentiment

#r-square for neg: 0.217 adjusted r-squared: 0.008            #RESULTS
#r-square for pos: 0.408 adjusted r-squared: 0.250
#r-square for mean_comp: 0.402 adjusted r-squared: 0.242

##
## 6. VISUALIZE
##

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)                      #Plotting negative sentiment and scores by round
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Negative Sentiment and Score by Round')
ax1.scatter(merged_df['neg1'], merged_df['Score_R1'])
ax1.set_title("Round 1")
ax2.scatter(merged_df['neg2'], merged_df['Score_R2'])
ax2.set_title("Round 2")
ax3.scatter(merged_df['neg3'], merged_df['Score_R3'])
ax3.set_title("Round 3")
ax4.scatter(merged_df['neg4'], merged_df['Score_R4'])
ax4.set_title("Round 4")

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)                      #Plotting negative sentiment and position by round
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Negative Sentiment and Position by Round')
ax1.scatter(merged_df['neg1'], merged_df['Pos_R1'])
ax1.set_title("Round 1")
ax2.scatter(merged_df['neg2'], merged_df['Pos_R2'])
ax2.set_title("Round 2")
ax3.scatter(merged_df['neg3'], merged_df['Pos_R3'])
ax3.set_title("Round 3")
ax4.scatter(merged_df['neg4'], merged_df['finish'])
ax4.set_title("Round 4")