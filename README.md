Following a rather heartbreaking third place finish in the 150th Open, golfer Rory McIlroy gave an interview stating how he is staying positive an a win is sure to follow. The statement was celebrated as a great example of how to remain resilient and the mindset it takes to be a successful sports person. Given that it is an individual sport and the slower pace compared to other sports, I wanted to see how much of an effect mindset has on performance in a sport like golf.

There is extensive research on how sports and activity improve mental health but far less on the converse. As someone who has had a varied career with many wins and years of dry spells, I chose to study golfer Rory McIroy. I used Natural Language Processing to understand mindset using sentiment analysis. I set up models to see if sentiment from post round transcripts is useful in understanding performance in major tournament from 2014 to 2022.

Datasets
○ ASAP Sports Website: This was used to get the post round transcripts for
McIlroy and was manually copied into a .csv file with the Tournament name
and date.
○ Golf Stats Website: This was used to get round statistics such as score,
ranking, finish.
● Packages: pandas (data manipulation, mutation, merging), re (regular expressions to
find interviewer names, splitting by delimiter, removing words in parentheses), statistics (mean), nltk (sentiment analysis scores), statsmodels (linear regression models), matplotlib (data visualizations)

Multiple regression was used to study the strength of relationship between negative, positive, and compound sentiment and the finish at the end of the tournament. 

finish = 259.82*neg1 + 25.31*neg2 + 179.75*neg3 + 149.75*neg4 - 14.0944
Negative sentiment scores seemed to have the lowest r-squared value. A model using golf scores following each round was used to see whether sentiment was just a proxy for score.

Negative sentiment was plotted against position and score for each round, there seems to be more of a linear relationship between sentiment and position rather than score.

For the model using negative sentiment, negativity score after round 1 (neg1) has the greatest effect, with a coefficient of 259.82, followed by round 3 (neg3) and round 4 (neg4) with coefficients 179.75 and 149.75 respectively, and lastly round 2 (neg2) with coefficient 25.31. This work is a proof of concept that post game transcripts are a rich source of player sentiment that have an effect on performance. Future work includes studying frequent n-grams in negatively scored texts to recognize these phrases. Additionally, further clarifying what constitutes negative sentiment and identifying what leads to bad performance, whether it is pessimism, hopelessness, etc. It would also be interesting to see if a similar trend is seen in other players on the PGA tour.
