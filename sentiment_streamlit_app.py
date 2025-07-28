import streamlit as st
from serpapi import GoogleSearch
import os
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# App title
st.title("🌍 Media Sentiment Explorer (via Google + Hugging Face)")

# API Key Input
api_key = os.getenv("SERPAPI_API_KEY")
if not api_key:
    api_key = st.text_input("🔑 Enter your SerpApi API Key", type="password")

# Search Input
search_term = st.text_input("🔎 Search Term", value="Jews")
location = st.text_input("📍 Location", value="New York, New York, United States")

# Search Trigger
if st.button("Run Search and Analyze"):

    if not api_key or not search_term:
        st.error("Please provide both an API key and a search term.")
    else:
        with st.spinner("Searching Google and analyzing sentiment..."):

            # Step 1: Google Search via SerpApi
            params = {
                "q": search_term,
                "location": location,
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "api_key": api_key
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            # Step 2: Save raw results (overwrite if needed)
            safe_term = "".join(c if c.isalnum() else "_" for c in search_term.lower())
            filename = f"search_{safe_term}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)

            # Step 3: Extract Text for Sentiment Analysis
            texts = []

            if "organic_results" in results:
                for item in results["organic_results"]:
                    if "title" in item:
                        texts.append(item["title"])
                    if "snippet" in item:
                        texts.append(item["snippet"])

            if "related_questions" in results:
                for question in results["related_questions"]:
                    texts.append(question.get("question", ""))
                    for block in question.get("text_blocks", []):
                        if "snippet" in block:
                            texts.append(block["snippet"])
                        if "list" in block:
                            for entry in block["list"]:
                                texts.append(entry.get("snippet", ""))

            # Step 4: Load Sentiment Model
            model_name = "tabularisai/multilingual-sentiment-analysis"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            def predict_sentiment(texts):
                inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
                return [sentiment_map[p] for p in torch.argmax(probs, dim=-1).tolist()]

            # Step 5: Analyze in Batches
            def batch_sentiment(texts, batch_size=20):
                results = []
                for i in range(0, len(texts), batch_size):
                    results.extend(predict_sentiment(texts[i:i + batch_size]))
                return results

            sentiments = batch_sentiment(texts)

            # Step 6: Display Results
            # st.subheader("📋 Individual Sentiment Results")
            # for text, sentiment in zip(texts, sentiments):
            #     st.markdown(f"**{sentiment}** — {text}")
            sentiment_score_map = {
                "Very Negative": 0,
                "Negative": 1,
                "Neutral": 2,
                "Positive": 3,
                "Very Positive": 4
            }

            st.subheader("\n📊 Sentiment Distribution:")
            
            # Step 7: Analyze Overall
            counts = Counter(sentiments)
            total = sum(counts.values())
            percentages = {k: (v / total * 100) for k, v in counts.items()}
            for sentiment, percent in percentages.items():
              st.markdown(f"{sentiment}: {percent:.2f} ({counts[sentiment]} occurrences)")
            avg_score = sum(sentiment_score_map[s] for s in sentiments) / len(sentiments)

            def interpret(score):
                if score >= 3.5:
                    return "Strongly Positive"
                elif score >= 2.5:
                    return "Moderately Positive"
                elif score >= 1.5:
                    return "Neutral to Slightly Negative"
                else:
                    return "Strongly Negative"

            overall = interpret(avg_score)

            st.markdown(f"**Overall Sentiment Score:** `{avg_score:.2f}`")
            st.markdown(f"**Interpretation:** `{overall}`")

            # Step 8: Visualization
            st.subheader("📊 Sentiment Breakdown")

            # fig, ax = plt.subplots()
            # ax.pie([counts[k] for k in counts], labels=counts.keys(), autopct='%1.1f%%', startangle=140)
            # ax.axis("equal")
            # st.pyplot(fig)
            # Custom color palette
            colors = ["#FCC6BB", "#FCF4BB", "#EABBFC", "#BBE2FC", "#C9FCBB"]  # red, yellow, purple, pruple, blue, green, blue
            sentiment_order = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

            # Match colors to sentiments found in the data
            labels = [s for s in sentiment_order if s in counts]
            sizes = [counts[s] for s in labels]
            matched_colors = [colors[sentiment_order.index(s)] for s in labels]

            # Create larger pie chart
            fig, ax = plt.subplots(figsize=(6, 6))  # <-- size in inches
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=matched_colors)
            ax.axis("equal")  # Equal aspect ratio ensures the pie is circular

            st.pyplot(fig)

            # Step 9: Optional Download
            sentiment_data = [{"text": t, "sentiment": s} for t, s in zip(texts, sentiments)]
            sentiment_filename = f"sentiment_{safe_term}.json"
            with open(sentiment_filename, "w") as f:
                json.dump(sentiment_data, f, indent=2)

            st.download_button(
                label="⬇️ Download Sentiment JSON",
                data=json.dumps(sentiment_data, indent=2),
                file_name=sentiment_filename,
                mime="application/json"
            )