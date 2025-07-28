# First run "source ~/.bashrc" in terminal
import os
from serpapi import GoogleSearch
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the API key from environment or text input
api_key = os.getenv("SERPAPI_API_KEY")

# Parameters for the API search query
params = {
  "q": "Jews",
  "location": "New York, New York, United States",
  "hl": "en",
  "gl": "us",
  "google_domain": "google.com",
  "api_key": api_key
}

# Search Google and load results
search = GoogleSearch(params)
results = search.get_dict()

# # Saving search results in new file
search_term = params["q"]
# filename = f"search_{search_term.lower()}.json"
# with open(filename, "w") as f:
#   json.dump(results, f, indent=2)

# Exporting search results to sentiment analysis tool 'Transformers' in HuggingFace
# Step 1: Extract relevant text snippers for sentiment analysis
texts = []

# From organic search results
if "organic_results" in results:
  for item in results["organic_results"]:
    if "title" in item:
      texts.append(item["title"])
    if "snippet" in item:
      texts.append(item["snippet"])

# From related questions and AI overview
if "related_questions" in results:
  for question in results["related_questions"]:
    texts.append(question.get("question", ""))
    for block in question.get("text_blocks", []):
      if "snippet" in block:
        texts.append(block["snippet"])
      if "list" in block:
        for entry in block["list"]:
          texts.append(entry.get("snippet", ""))

# Step 2: Set up HuggingFace model for sentiment analysis
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(texts):
  inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
  with torch.no_grad():
    outputs = model(**inputs)
  probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
  sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
  return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

# Step 3: Run analysis in batches
def batch_sentiment(texts, batch_size=20):
  results = []
  for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results.extend(predict_sentiment(batch))
  return results

# Step 4: Analyze and print the sentiments
sentiments = batch_sentiment(texts)

# # Print each text snippet and sentiment rating
# for text, sentiment in zip(texts, sentiments):
#     print(f"Text: {text}\nSentiment: {sentiment}\n")

# Define sentiment mapping to numeric values
sentiment_score_map = {
    "Very Negative": 0,
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
    "Very Positive": 4
}

# Count how often each sentiment appears
sentiment_counts = Counter(sentiments)

# Total number of predictions
total = sum(sentiment_counts.values())

# Compute percentages
percentages = {k: f"{(v / total * 100):.1f}%" for k, v in sentiment_counts.items()}

# Compute average sentiment score
numeric_scores = [sentiment_score_map[s] for s in sentiments]
average_score = sum(numeric_scores) / len(numeric_scores)

# Interpret overall sentiment
def interpret_score(avg):
  if avg >= 3.5:
    return "Strongly Positive"
  elif avg >= 2.5:
    return "Moderately Positive"
  elif avg >= 1.5:
    return "Neutral to Slightly Negative"
  else:
    return "Strongly Negative"

overall_sentiment = interpret_score(average_score)

# Output the results
print("\n📊 Sentiment Distribution:")
for sentiment, percent in percentages.items():
  print(f"{sentiment}: {percent} ({sentiment_counts[sentiment]} occurrences)")

print(f"\n📈 Overall Sentiment Score: {average_score:.2f}")
print(f"🧠 Overall Interpretation: {overall_sentiment}")

# Optional: Save to file
sentiment_file = f"sentiment_{search_term.lower()}.json"
with open(sentiment_file, "w") as f:
  json.dump([{"text": t, "sentiment": s} for t, s in zip(texts, sentiments)], f, indent=2)

print(f"\n✅ Sentiment analysis saved to '{sentiment_file}'")

# @misc{tabularisai_2025,
#     author       = { tabularisai and Samuel Gyamfi and Vadim Borisov and Richard H. Schreiber },
#     title        = { multilingual-sentiment-analysis (Revision 69afb83) },
#     year         = 2025,
#     url          = { https://huggingface.co/tabularisai/multilingual-sentiment-analysis },
#     doi          = { 10.57967/hf/5968 },
#     publisher    = { Hugging Face }
# }