import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import openai
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .sentiment-positive {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    }
    
    .sentiment-negative {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
    }
    
    .sentiment-neutral {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
    }
    
    .analysis-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        color: #1f2937;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-size: 16px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    .positive-badge {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .negative-badge {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .neutral-badge {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fed7aa;
    }
</style>
""", unsafe_allow_html=True)

# Download necessary NLTK data if not already downloaded
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

download_nltk_data()

# Set up OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🔑 OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")
    st.stop()

openai.api_key = openai_api_key

# Preprocessing function (must be identical to the one used in train.py)
def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove # symbol
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]  # Stemming and stopword removal
    return ' '.join(tokens)

# Function to get detailed analysis from OpenAI
@st.cache_data
def get_detailed_analysis(text, predicted_sentiment):
    prompt = f"Given the text: \"{text}\", and its predicted sentiment is {predicted_sentiment.lower()}. Provide a comprehensive and detailed analysis of the text, explaining why it might have this sentiment, and discuss any nuanced aspects or potential interpretations, especially considering its context in social media like Twitter. If it's a neutral sentiment, elaborate on what specific aspects make it neutral and how different people might perceive it. Keep the response concise but informative, around 2-3 paragraphs."
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes social media text for sentiment and provides detailed explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        st.error(f"OpenAI API Error: {e}")
        return "Could not generate detailed analysis due to an API error."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "Could not generate detailed analysis."

# Load the trained model pipeline
@st.cache_resource
def load_model():
    model_path = 'social-media-nlp/model/sentiment_pipeline.joblib'
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error("❌ Model not found! Please run train.py first to train and save the model.")
        st.stop()

# Create confidence chart
def create_confidence_chart(probabilities, class_labels):
    # Color mapping for sentiments
    colors = {
        'positive': '#10b981',
        'negative': '#ef4444', 
        'neutral': '#f59e0b'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=[label.capitalize() for label in class_labels],
            y=[prob * 100 for prob in probabilities],
            marker_color=[colors.get(label, '#667eea') for label in class_labels],
            text=[f'{prob:.1%}' for prob in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Sentiment Confidence Scores',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        xaxis_title="Sentiment",
        yaxis_title="Confidence (%)",
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_layout(yaxis=dict(range=[0, 100]))
    
    return fig

# Header
st.markdown("""
<div class="main-header">
    <h1>🐦 Twitter Sentiment Analyzer</h1>
    <p>AI-Powered Social Media Text Analysis with Advanced NLP</p>
</div>
""", unsafe_allow_html=True)

# Load model
pipeline = load_model()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Text")
    
    # Sample texts for quick testing
    st.markdown("**Quick Test Examples:**")
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("😊 Positive Example", use_container_width=True):
            st.session_state.sample_text = "I absolutely love this new feature! It's amazing and works perfectly!"
            
    with examples_col2:
        if st.button("😐 Neutral Example", use_container_width=True):
            st.session_state.sample_text = "The weather today is okay, nothing special to report."
            
    with examples_col3:
        if st.button("😞 Negative Example", use_container_width=True):
            st.session_state.sample_text = "This service is terrible and I'm really disappointed with the experience."
    
    user_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('sample_text', ''),
        height=120,
        placeholder="Type or paste your social media text here for sentiment analysis..."
    )
    
    analyze_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    <div class="feature-highlight">
        <strong>✨ Key Features</strong><br>
        • Advanced NLP preprocessing<br>
        • Machine Learning predictions<br>
        • AI-powered detailed analysis<br>
        • Real-time confidence scores
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **How it works:**
    1. Text preprocessing (cleaning, tokenization)
    2. Logistic Regression classification
    3. OpenAI GPT analysis for insights
    4. Visual confidence reporting
    """)

# Analysis Results
if analyze_button:
    if user_input:
        with st.spinner("🔄 Processing your text..."):
            # Preprocess the input text
            processed_input = preprocess_text(user_input)
            
            if not processed_input.strip():
                st.warning("⚠️ The processed text is empty. Please try a different input.")
            else:
                # Get prediction probabilities
                probabilities = pipeline.predict_proba([processed_input])[0]
                class_labels = pipeline.classes_
                
                # Predict the sentiment
                predicted_sentiment = pipeline.predict([processed_input])[0]
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Main prediction with styled badge
                sentiment_classes = {
                    'positive': ('positive-badge', '😊'),
                    'negative': ('negative-badge', '😞'),
                    'neutral': ('neutral-badge', '😐')
                }
                
                badge_class, emoji = sentiment_classes.get(predicted_sentiment, ('neutral-badge', '😐'))
                
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <h3>Predicted Sentiment</h3>
                    <div class="prediction-badge {badge_class}">
                        {emoji} {predicted_sentiment.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores visualization
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Create and display the chart
                    fig = create_confidence_chart(probabilities, class_labels)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 Confidence Breakdown")
                    
                    # Create metrics for each sentiment
                    sentiment_order = {'negative': 0, 'neutral': 1, 'positive': 2}
                    ordered_data = sorted(zip(class_labels, probabilities), key=lambda x: sentiment_order[x[0]])
                    
                    for sentiment, prob in ordered_data:
                        emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                        st.metric(
                            label=f"{emoji_map[sentiment]} {sentiment.capitalize()}",
                            value=f"{prob:.1%}",
                            delta=f"{'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low'} confidence"
                        )
                
                # Model information
                st.info("🤖 **Model Info:** This Logistic Regression model was trained on Twitter data to understand social media language patterns and context.")
                
                # Detailed AI Analysis
                st.markdown("### 🧠 AI-Powered Detailed Analysis")
                
                with st.spinner("🤖 Generating comprehensive analysis..."):
                    detailed_analysis = get_detailed_analysis(user_input, predicted_sentiment)
                    
                    st.markdown(f"""
                    <div class="analysis-box">
                        <h4>📋 Expert Analysis</h4>
                        <p>{detailed_analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                max_confidence = max(probabilities)
                if max_confidence < 0.6:
                    st.warning("⚠️ **Low Confidence Alert:** The model shows mixed signals. This text might have ambiguous sentiment or contain conflicting emotional cues.")
                elif max_confidence > 0.8:
                    st.success("✅ **High Confidence:** The model is very confident about this prediction based on clear sentiment indicators.")
    
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p>Built with ❤️ using Streamlit, NLTK, OpenAI, and Plotly | 
    <strong>Twitter Sentiment Analyzer v2.0</strong></p>
    <p><em>Powered by Machine Learning and AI for accurate social media sentiment analysis</em></p>
</div>
""", unsafe_allow_html=True)