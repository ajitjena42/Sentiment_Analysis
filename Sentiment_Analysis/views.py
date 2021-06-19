from django.shortcuts import render
from django.http import HttpResponse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud,STOPWORDS
from nltk import tokenize
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nltk
import re
import string

# Create your views here.
SIA=SentimentIntensityAnalyzer()
sum_=1
def clean(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def tokenization(a):
    return tokenize.word_tokenize(a)

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

def percentage(k):
    return (k/sum_)*100

def nielit(request):
    if request.method=="POST":
        name = request.POST.get("path","")
        df=pd.read_csv(name)
        df['text'] = df['text'].apply(lambda x:clean(x))
        scores=[]
        for i in range(len(df['text'])):
            score = SIA.polarity_scores(df['text'][i])
            score=score['compound']
            scores.append(score)
        sentiment=[]
        for i in scores:
            if i>=0.05:
                sentiment.append('Positive')
            elif i<=(-0.05):
                sentiment.append('Negative')
            else:
                sentiment.append('Neutral')
        df['sentiment']=pd.Series(np.array(sentiment))
        df['text'] = df['text'].apply(lambda x:clean_text(x))

        tweets=df[['text','sentiment']].drop_duplicates()
        tweets['tokenized'] = tweets['text'].apply(lambda x: tokenization(x.lower()))
        stopword = nltk.corpus.stopwords.words('english')
        stopword.extend(['covid','vaccine','vaccines','dose','doses','vaccinated','vaccination','pfizer','covidvaccine','amp','first','today','corona','coronavirus','got','day','covidvaccination','second','take','took','jab','jabs','pfizerbiontech'])

        def remove_stopwords(text):
            text = [word for word in text if word not in stopword]
            return text
        tweets['No_stopwords'] = tweets['tokenized'].apply(lambda x: remove_stopwords(x))

        #Pie_Chart
        tags=tweets['sentiment'].value_counts().sort_values(ascending=False)
        tags=dict(tags)
        
        pos_=tags['Positive']
        neg_=tags['Negative']
        neu_=tags['Neutral']
        sum_=pos_+neg_+neu_
        def percentage(k):
            return (k/sum_)*100
        slices_tweets = [percentage(pos_), percentage(neu_),percentage(neg_)]
        
        analysis = ['Positive', 'Neutral', 'Negative']
        colors = ['g', 'y', 'r']
        plt.pie(slices_tweets, labels=analysis,explode = (0.03, 0.03, 0.03) ,shadow=False, autopct='%1.1f%%') 
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\pie_chart.png',format='png',dpi=600)
        matplotlib.pyplot.clf()
        plt.bar(analysis,[pos_,neu_,neg_],color=colors,width=0.5)
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\bar.png',format='png',dpi=600)
        matplotlib.pyplot.clf()
        plt.plot(analysis, [pos_,neu_,neg_])
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\plot.png',format='png',dpi=600)
        matplotlib.pyplot.clf()
        temp=tweets.groupby('sentiment')
        def merge(x):
            return ' '.join(i for i in x)

        temp1=temp.get_group('Positive').No_stopwords.apply(lambda x: merge(x))
        temp2=temp.get_group('Negative').No_stopwords.apply(lambda x: merge(x))
        temp3=temp.get_group('Neutral').No_stopwords.apply(lambda x: merge(x))

        #Wordcloud of Positive Tweets
        text1 = ",".join(review for review in temp1)
        wordcloud = WordCloud(width=2000, height=2000,max_words=100, colormap='Set2',background_color="black",stopwords=STOPWORDS).generate(text1)
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\positive_wordcloud.png',format='png',dpi=600)
        matplotlib.pyplot.clf()

        #Wordcloud of Negative Tweets
        text2 = ",".join(review for review in temp2)
        wordcloud = WordCloud(width=2000, height=2000, max_words=100, colormap='Set2',background_color="black",stopwords=STOPWORDS).generate(text2)
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\negative_wordcloud.png',format='png',dpi=600)
        matplotlib.pyplot.clf()

        #Wordcloud of Neutral Tweets
        text3 = ",".join(review for review in temp3)
        wordcloud = WordCloud(width=2000, height=2000, max_words=100, colormap='Set2',background_color="black",stopwords=STOPWORDS).generate(text3)
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\neutral_wordcloud.png',format='png',dpi=300)
        matplotlib.pyplot.clf()

        #Wordcloud of all Tweets
        text4 = text1+' '+text2+' '+text3
        wordcloud = WordCloud(width=2000, height=2000, max_words=100, colormap='Set2',background_color="black",stopwords=STOPWORDS).generate(text4)
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(r'A:\Desktop\NIELIT_COVID19_TWEETS\Sentiment_Analysis\static\img\all_tweets_wordcloud.png',format='png',dpi=300)
        matplotlib.pyplot.clf()

        data=df.to_html()
        data=data.replace('class="dataframe"','class="table table-responsive table-striped table-bordered " style="width:100%" id="datatable"')
        saveSuccess=True
        return render(request, "analysis.html", {"saveSuccess":saveSuccess,"data":data})
    saveSuccess=False
    data=" "
    return render(request, "analysis.html", {"saveSuccess":saveSuccess,"data":data})