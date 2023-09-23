import streamlit as st
import pandas as pd
import numpy as np
import dill as pickle
import helper_functions as hf


st.set_page_config(
    page_title="Amazon Sentiment Analyzer", page_icon="📊", layout="wide"
)


adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

def search_callback():
    # Initialize dataset
    st.session_state.df = pd.read_excel('amazon_reviews.xlsx', index_col=0).sample(4000)
    with open('logistic_regression_model.pkl', 'rb') as f:
        st.session_state.lr_loaded = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        st.session_state.cvec_loaded = pickle.load(f)

with st.sidebar:
    st.title("Amazon Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the latest Amazon reviews.
            Since the app can only predict positive or 
            negative sentiment, it is more suitable towards analyzing the 
            sentiment of brand, product, service, company, or person. 
        </div>
        """,
        unsafe_allow_html=True,
    )


    search_callback()

def predict_sentiment(user_input):
    transformed_input = st.session_state.cvec_loaded.transform([user_input])
    prediction = st.session_state.lr_loaded.predict_proba(transformed_input)
    label = np.argmax(prediction)
    return ('Negative', prediction[0][label]) if label == 0 else ('Positive', prediction[0][label])

if "df" in st.session_state:

    def make_dashboard(df, bar_color, wc_color):
    
        
        # Create a function to format the boxes
        def create_stat_box(stat_name, value):
            return f"""
            <div style="border: 1px solid #ddd; padding: 10px; width: 70%; margin: 0 auto;">
                <p style="margin-bottom: 5px;"><strong>{stat_name}</strong></p>
                <p style="margin: 0;">{value}</p>
            </div>
            """
        # first row
        col1, col2, col3 = st.columns([28, 34, 38])
        with col1:
            sentiment_plot = hf.plot_sentiment(df)
            sentiment_plot.update_layout(height=350, title_x=0.5)
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        with col2:
            top_unigram = hf.get_top_n_gram(df, ngram_range=1, n=10)
            unigram_plot = hf.plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
            unigram_plot.update_layout(height=350)
            st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
        with col3:
            top_bigram = hf.get_top_n_gram(df, ngram_range=2, n=10)
            bigram_plot = hf.plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
            bigram_plot.update_layout(height=350)
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

        # second row
        col1, col2 = st.columns([60, 40])
        with col1:

            def sentiment_color(sentiment):
                if sentiment == "Positive":
                    return "background-color: #1F77B4; color: white"
                else:
                    return "background-color: #FF7F0E"
            st.dataframe(
                df[["sentiment", "text"]].sample(100).style.applymap(
                    sentiment_color, subset=["sentiment"]
                ),
                height=575,
                use_container_width=True
            )
        with col2:
            wordcloud = hf.plot_wordcloud(df, colormap=wc_color)
            st.pyplot(wordcloud)

    adjust_tab_font = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    </style>
    """

    st.write(adjust_tab_font, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["All", "Positive 😊", "Negative ☹️", "Prediction"])
    with tab1:
        make_dashboard(st.session_state.df, bar_color="#54A24B", wc_color="Greens")
    with tab2:
        dff = st.session_state.df[st.session_state.df['sentiment'] == 'Positive']
        make_dashboard(dff, bar_color="#1F77B4", wc_color="Blues")
    with tab3:
        dff = st.session_state.df[st.session_state.df['sentiment'] == 'Negative']
        make_dashboard(dff, bar_color="#FF7F0E", wc_color="Oranges")
    with tab4:
        st.title('Sentiment Analysis')
        
        # Add some padding around the text area for aesthetics
        st.write("<style>textarea { padding: 1em; }</style>", unsafe_allow_html=True)
        
        user_input = st.text_area("Enter your text for sentiment analysis:", height=200)
        
        if st.button('Analyze Sentiment'):
            prediction, probability = predict_sentiment(user_input)
            if prediction == "Positive":
                st.markdown(f'<p style="font-size: 18px; color:green;">Predicted Sentiment: <strong>{prediction}</strong> with probability {probability:.2%}</p>', unsafe_allow_html=True)
            elif prediction == "Negative":
                st.markdown(f'<p style="font-size: 18px; color:red;">Predicted Sentiment: <strong>{prediction}</strong> with probability {probability:.2%}</p>', unsafe_allow_html=True)
            else: # Neutral or other classes
                st.markdown(f'<p style="font-size: 18px; color:blue;">Predicted Sentiment: <strong>{prediction}</strong> with probability {probability:.2%}</p>', unsafe_allow_html=True)
