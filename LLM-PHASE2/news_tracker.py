import streamlit as st
import ollama
from datetime import datetime, timedelta
import json
import feedparser
import requests
import time
from urllib.parse import quote

# --- App Configuration ---
st.set_page_config(
    page_title="Global News Topic Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global News Topic Tracker")
st.write("Track trending topics from Google News and get AI-powered summaries using local LLMs")

# --- Session State Initialization ---
if "news_data" not in st.session_state:
    st.session_state.news_data = None
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("✅ **100% Local Processing**\n\nUsing Google News + Ollama\nNo API keys required!")
    
    st.divider()
    
    st.write("**News Settings**")
    
    # Region selection
    region = st.selectbox(
        "Region:",
        options=["US", "UK", "CA", "AU", "IN", "DE", "FR", "IT", "ES", "BR", "MX", "JP", "KR", "CN"],
        index=0,
        help="Select the region for news",
        key="region"
    )
    
    # Language selection
    language = st.selectbox(
        "Language:",
        options=["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
        index=0,
        help="Select the language for news",
        key="language"
    )
    
    # Number of articles
    num_articles = st.slider(
        "Number of Articles:",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of articles to fetch per topic",
        key="num_articles"
    )
    
    st.divider()
    
    st.write("**Ollama Model Selection**")
    
    # Try to get available Ollama models
    try:
        available_models = ollama.list()
        model_names = [model['name'] for model in available_models.get('models', [])]
        
        # Filter for text models (not vision)
        text_models = [m for m in model_names if 'llava' not in m.lower()]
        
        if text_models:
            ollama_model_options = text_models
            # Prefer smaller/faster models
            preferred_order = ["mistral:latest", "llama3:latest", "gpt-oss:20b"]
            for pref in preferred_order:
                if pref in ollama_model_options:
                    default_ollama_index = ollama_model_options.index(pref)
                    break
            else:
                default_ollama_index = 0
        else:
            ollama_model_options = ["mistral:latest", "llama3:latest"]
            default_ollama_index = 0
            st.warning("No Ollama models found. Install one with: `ollama pull mistral:latest`")
    except Exception as e:
        ollama_model_options = ["mistral:latest", "llama3:latest"]
        default_ollama_index = 0
        st.warning("Could not connect to Ollama. Make sure it's running.")
    
    ollama_model = st.selectbox(
        "Ollama Model for Summarization:",
        options=ollama_model_options,
        index=default_ollama_index,
        help="Local Ollama model for summarizing news topics",
        key="ollama_model"
    )
    
    st.divider()
    
    st.write("**Summary Options**")
    summarize_topics = st.checkbox("Summarize Topics", value=True, key="summarize_topics")
    extract_key_points = st.checkbox("Extract Key Points", value=True, key="extract_key_points")
    analyze_sentiment = st.checkbox("Analyze Sentiment", value=False, key="analyze_sentiment")
    
    st.divider()
    
    if st.button("🗑️ Clear All"):
        st.session_state.news_data = None
        st.session_state.summaries = {}
        st.session_state.selected_topic = None
        st.rerun()

# --- Main Content ---
st.write("### 📰 Fetch News")

# Search options
col1, col2 = st.columns([3, 1])

with col1:
    search_query = st.text_input(
        "Search Topic:",
        placeholder="e.g., Artificial Intelligence, Climate Change, Technology",
        value=st.session_state.get('search_query', ''),
        key="search_query"
    )

with col2:
    fetch_button = st.button("🔍 Fetch News", type="primary", use_container_width=True)

# Quick topic buttons
st.write("**Quick Topics:**")
topic_buttons = st.columns(5)
quick_topics = ["Technology", "AI", "Climate", "Politics", "Health"]

search_triggered = False
for i, topic in enumerate(quick_topics):
    with topic_buttons[i]:
        if st.button(topic, key=f"topic_{topic}", use_container_width=True):
            search_query = topic
            st.session_state.search_query = topic
            search_triggered = True

# --- Fetch and Process News ---
# Use search_query from state if available, otherwise use input
current_query = st.session_state.get('search_query', search_query) if search_query else st.session_state.get('search_query', '')

if fetch_button or search_triggered:
    if not current_query:
        st.warning("⚠️ Please enter a search topic")
    else:
        with st.spinner(f"🔍 Fetching news about '{current_query}' from {st.session_state.region}..."):
            try:
                # Build Google News RSS URL
                query_encoded = quote(current_query)
                # Google News RSS feed URL
                rss_url = f"https://news.google.com/rss/search?q={query_encoded}&hl={st.session_state.language}&gl={st.session_state.region}&ceid={st.session_state.region}:{st.session_state.language}"
                
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                # Extract articles
                articles = []
                for entry in feed.entries[:st.session_state.num_articles]:
                    # Extract source from title (Google News format: "Title - Source")
                    title_parts = entry.get('title', '').split(' - ')
                    if len(title_parts) > 1:
                        title = ' - '.join(title_parts[:-1])
                        source = title_parts[-1]
                    else:
                        title = entry.get('title', '')
                        source = entry.get('source', {}).get('title', 'Unknown') if hasattr(entry, 'source') else 'Unknown'
                    
                    articles.append({
                        'title': title,
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source,
                        'summary': entry.get('summary', '') or entry.get('description', '')
                    })
                
                if articles:
                    st.session_state.news_data = {
                        'query': current_query,
                        'articles': articles,
                        'fetched_at': datetime.now().isoformat(),
                        'region': st.session_state.region,
                        'language': st.session_state.language
                    }
                    st.success(f"✅ Found {len(articles)} articles about '{current_query}'")
                    st.rerun()
                else:
                    st.warning("⚠️ No articles found. Try a different search term.")
                    
            except Exception as e:
                st.error(f"❌ Error fetching news: {str(e)}")
                st.info("💡 Make sure you have internet connection and required packages are installed: `pip install feedparser beautifulsoup4 dateparser requests`")

# --- Display News Articles ---
if st.session_state.news_data:
    st.divider()
    st.header(f"📊 News about: {st.session_state.news_data['query']}")
    
    # Summary section
    if st.session_state.summarize_topics and st.session_state.news_data['query'] not in st.session_state.summaries:
        if st.button("🤖 Generate AI Summary", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing news and generating summary..."):
                try:
                    # Prepare articles text for summarization
                    articles_text = "\n\n".join([
                        f"**{i+1}. {article['title']}**\n"
                        f"Source: {article['source']}\n"
                        f"Summary: {article['summary']}\n"
                        f"Link: {article['link']}"
                        for i, article in enumerate(st.session_state.news_data['articles'])
                    ])
                    
                    # Build prompt
                    prompt_parts = [f"Analyze the following {len(st.session_state.news_data['articles'])} news articles about '{st.session_state.news_data['query']}':\n\n"]
                    
                    if st.session_state.extract_key_points:
                        prompt_parts.append("Extract the key points and main themes.\n")
                    if st.session_state.analyze_sentiment:
                        prompt_parts.append("Analyze the overall sentiment (positive, negative, neutral).\n")
                    
                    prompt_parts.append("Provide a comprehensive summary in JSON format:\n")
                    prompt_parts.append("""{
    "overall_summary": "A brief overall summary of the news topic",
    "key_themes": ["Theme 1", "Theme 2", "Theme 3"],
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "trending_aspects": ["Aspect 1", "Aspect 2"],
    "sentiment": "positive/negative/neutral/mixed",
    "significance": "Why this topic is trending or important"
}""")
                    
                    prompt_parts.append(f"\n\nArticles:\n{articles_text}\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no code blocks, just the raw JSON object.")
                    
                    prompt = "".join(prompt_parts)
                    
                    # Call Ollama
                    response = ollama.chat(
                        model=st.session_state.ollama_model,
                        messages=[
                            {"role": "system", "content": "You are a news analyst. Analyze news articles and provide structured summaries in JSON format. Always return valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        options={"temperature": 0.3}
                    )
                    
                    # Parse response
                    response_text = response['message']['content'].strip()
                    
                    # Try to extract JSON
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    # Parse JSON
                    try:
                        summary_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Try to find JSON in text
                        import re
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            summary_data = json.loads(json_match.group())
                        else:
                            # Fallback: create summary from text
                            summary_data = {
                                "overall_summary": response_text[:500],
                                "key_themes": [],
                                "key_points": [],
                                "trending_aspects": [],
                                "sentiment": "neutral",
                                "significance": ""
                            }
                    
                    st.session_state.summaries[st.session_state.news_data['query']] = summary_data
                    st.success("✅ Summary generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    st.info("💡 Make sure Ollama is running and the model is installed.")
    
    # Display summary if available
    if st.session_state.news_data['query'] in st.session_state.summaries:
        summary = st.session_state.summaries[st.session_state.news_data['query']]
        
        st.divider()
        st.header("🤖 AI Analysis & Summary")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔑 Key Themes", "📈 Trending Aspects", "📄 Full Analysis"])
        
        with tab1:
            if summary.get("overall_summary"):
                st.write("### Overall Summary")
                st.info(summary["overall_summary"])
            
            col1, col2 = st.columns(2)
            with col1:
                if summary.get("sentiment"):
                    sentiment_emoji = {
                        "positive": "😊",
                        "negative": "😞",
                        "neutral": "😐",
                        "mixed": "😕"
                    }.get(summary["sentiment"].lower(), "😐")
                    st.metric("Sentiment", f"{sentiment_emoji} {summary['sentiment'].capitalize()}")
            
            with col2:
                if summary.get("significance"):
                    st.write("### Significance")
                    st.caption(summary["significance"])
        
        with tab2:
            if summary.get("key_themes"):
                st.write("### Key Themes")
                for i, theme in enumerate(summary["key_themes"], 1):
                    st.markdown(f"{i}. **{theme}**")
            
            if summary.get("key_points"):
                st.write("### Key Points")
                for i, point in enumerate(summary["key_points"], 1):
                    st.markdown(f"• {point}")
        
        with tab3:
            if summary.get("trending_aspects"):
                st.write("### Trending Aspects")
                for aspect in summary["trending_aspects"]:
                    st.success(f"🔥 {aspect}")
        
        with tab4:
            st.write("### Full Analysis Data")
            st.json(summary)
            
            # Download JSON
            json_str = json.dumps(summary, indent=2)
            st.download_button(
                "📥 Download Summary",
                json_str,
                file_name=f"news_summary_{st.session_state.news_data['query'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.divider()
    st.header("📰 Articles")
    
    # Display articles
    for i, article in enumerate(st.session_state.news_data['articles'], 1):
        with st.expander(f"📄 {i}. {article['title']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Source:** {article['source']}")
                if article.get('published'):
                    try:
                        pub_date = datetime.fromisoformat(article['published'].replace('Z', '+00:00'))
                        st.caption(f"Published: {pub_date.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.caption(f"Published: {article['published']}")
            with col2:
                if article['link']:
                    st.link_button("🔗 Read Article", article['link'])
            
            if article.get('summary'):
                st.write("**Summary:**")
                st.write(article['summary'])
    
    # Download articles as JSON
    st.divider()
    articles_json = json.dumps(st.session_state.news_data, indent=2, default=str)
    st.download_button(
        "📥 Download All Articles (JSON)",
        articles_json,
        file_name=f"news_{st.session_state.news_data['query'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

else:
    st.info("👆 Enter a topic above or click a quick topic button to get started")
    
    # Instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        ### ✅ Features:
        
        - **100% Local Processing** - Uses Google News + Ollama (no API keys)
        - **Region & Language Support** - Get news from different countries
        - **AI-Powered Summaries** - Get intelligent analysis of trending topics
        - **Key Points Extraction** - Identify main themes and trends
        - **Sentiment Analysis** - Understand the tone of news coverage
        
        ### Steps:
        1. **Enter a search topic** or click a quick topic button
        2. **Click "Fetch News"** to get recent articles
        3. **Click "Generate AI Summary"** to analyze the news
        4. **Review articles** and download results
        
        ### Tips:
        - Use specific topics for better results
        - Adjust number of articles based on your needs
        - Try different regions for global perspectives
        - The AI summary helps identify trending aspects
        """)

