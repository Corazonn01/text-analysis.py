#based on the previous code (of cleaning) we are going to analyze the text 
# 1) Word Frequency Analysis (Counting how often each word appears in a text). 

from collections import Counter

#Count the frequency of each word in the filtered text
word_freq = Counter(preprocessed['filtered_words'])

print("Word Frequencies:", word_freq)

# 2) Sentiment Analysis (Determining if the sentiment of a text is positive, negative, or neutral).

from nltk.sentiment import SentimentIntensityAnalyzer

#Download the VADER lexicon
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

#Analyze the sentiment of the original text
sentiment = analyze_sentiment(your_text)

print("Sentiment Analysis:", sentiment)

# 3) Part-of-Speech Tagging

nltk.download('averaged_perceptron_tagger')

def tag_parts_of_speech(text):
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)

#Tag parts of speech in the original text
pos_tags = tag_parts_of_speech(your_text)

print("Part-of-Speech Tags:", pos_tags)
