import numpy as np
import pandas as pd
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from pythainlp.corpus import thai_stopwords
from nltk.util import ngrams
from collections import Counter
stopwords = list(thai_stopwords())

# Create a custom plotly theme and set it as default
pio.templates["custom"] = pio.templates["plotly_white"]
pio.templates["custom"].layout.margin = {"b": 25, "l": 25, "r": 25, "t": 50}
pio.templates["custom"].layout.width = 600
pio.templates["custom"].layout.height = 450
pio.templates["custom"].layout.autosize = False
pio.templates["custom"].layout.font.update(
    {"family": "Arial", "size": 12, "color": "#707070"}
)
pio.templates["custom"].layout.title.update(
    {
        "xref": "container",
        "yref": "container",
        "x": 0.5,
        "yanchor": "top",
        "font_size": 16,
        "y": 0.95,
        "font_color": "#353535",
    }
)
pio.templates["custom"].layout.xaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.yaxis.update(
    {"showline": True, "linecolor": "lightgray", "title_font_size": 14}
)
pio.templates["custom"].layout.colorway = [
    "#1F77B4",
    "#FF7F0E",
    "#54A24B",
    "#D62728",
    "#C355FA",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#FFE323",
    "#17BECF",
]
pio.templates.default = "custom"

def plot_sentiment(df):
    sentiment_count = df["sentiment"].value_counts()
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_wordcloud(df, colormap="Greens"):

    cmap = mpl.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:15])
    mask = np.array(Image.open("like_mask.png")) 
    mask = np.where(mask>0, 255, 0)
    words = " ".join(text for text in df['text'] if isinstance(text, str))
    wc = WordCloud(
        background_color="white",
        stopwords=stopwords,
        max_words=2000,
        # height = 2000,
        # width=4000,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )
    wc.generate(words)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig



def get_top_n_gram(df, ngram_range, n=10):
    # Tokenization using pythainlp
    words = " ".join(text for text in df['text'] if isinstance(text, str))
    tokens = words.split(" ")

    # Extract bigrams (2-grams)
    bigrams = list(ngrams(tokens, ngram_range))

    # Count the bigrams
    bigram_counts = Counter(bigrams)
    # Convert to DataFrame
    df = pd.DataFrame([( ' '.join(k), v) for k, v in bigram_counts.items()], columns=['words', 'counts'])
    df = df.sort_values('counts', ascending=False).reset_index(drop=True).head(n)

    return df

def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title=f"<b>{title}</b>",
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig

# def plot_line(facebook_df):
#     dff = facebook_df.groupby([facebook_df.time.dt.date, 'sentiment']).size().reset_index(name='count')
#     dff['time'] = pd.to_datetime(dff['time'])

#     # Plotly Express chart
#     fig = px.line(dff, x='time', y='count', color='sentiment',
#                 labels={'count': 'Number of Comments', 'time': 'Date'},
#                 title="Number of Comments over Time")
    
#     return fig

def plot_bar(df):
    # Count the occurrences of each sentiment
    dff = df['sentiment'].value_counts().reset_index(name='count')

    sentiments = dff['sentiment']
    counts = dff['count']

    fig = go.Figure()

    # Creating a bar chart
    fig.add_trace(go.Bar(
        x=sentiments,
        y=counts,
        text=counts,
        textposition='auto',
        textfont_size=18
    ))

    # Finalizing layout
    fig.update_layout(
        title='Sentiment Analysis Counts',
    )
    
    return fig






