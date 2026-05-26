# Dataset: Airline Sentiment on Twitter

## Source
- **Original competition**: [Twitter Airline Sentiment (CrowdFlower)](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- **File name**: `Tweets.csv`
- **Format**: CSV, UTF-8 encoding
- **License**: Check Kaggle page for usage terms

## Column Descriptions

| Column | Type | Description | Usage in Project |
|--------|------|-------------|-----------------|
| `tweet_id` | int64 | Unique identifier for each tweet | Metadata, deduplication |
| `airline_sentiment` | str | Label: `positive`, `negative`, or `neutral` | **Target variable** for classification |
| `airline_sentiment_confidence` | float | Model confidence in sentiment label [0.0–1.0] | **Confidence-aware training**: filter noisy labels, sample weights |
| `negativereason` | str | Reason for negative sentiment (e.g., "Late Flight", "Customer Service Issue") | Optional: feature engineering, multi-task learning |
| `negativereason_confidence` | float | Confidence in negative reason assignment | Optional: weight negative samples |
| `airline` | str | Airline name (e.g., "Virgin America", "United") | Optional: stratified analysis, one-hot feature |
| `airline_sentiment_gold` | str | Expert-verified label (if available) | Optional: high-quality subset evaluation |
| `name` | str | Twitter username of the author | PII — exclude from modeling |
| `negativereason_gold` | str | Expert-verified negative reason | Optional: validation of reason labels |
| `retweet_count` | int | Number of retweets | Optional: engagement analysis, weighting popular tweets |
| `text` | str | Full tweet text | **Primary input feature** for NLP pipeline |
| `tweet_coord` | str | Geo-coordinates `[lat, lon]` (if available) | Optional: geographic visualization |
| `tweet_created` | datetime | Timestamp of tweet (UTC-8) | Optional: temporal patterns, time-based splits |
| `tweet_location` | str | User-provided location string | Optional: geographic feature (noisy, many NAs) |
| `user_timezone` | str | User's timezone setting | Optional: regional analysis |

## 🔍 Key Observations & Interpretation

### 1. Target Variable: `airline_sentiment`
- **Classes**: `positive`, `negative`, `neutral`
- **Typical distribution**: ~60% negative, ~20% neutral, ~20% positive (imbalance expected)
- **Recommendation**: Use `class_weight='balanced'` or stratified sampling during train/test split

### 2. Confidence Columns: `*_confidence`
- Values range from ~0.3 to 1.0
- Lower confidence (~0.3–0.6) often indicates ambiguous or sarcastic tweets
- **Strategy**:
  - Filter: exclude samples with `confidence < 0.7` for high-precision training
  - Weight: use `sample_weight=confidence` in `LogisticRegression.fit()` to prioritize reliable labels

### 3. Text Column: `text`
- Contains mentions (`@VirginAmerica`), URLs, emojis, informal language
- **Preprocessing recommendations**:
  - Lowercasing
  - Remove URLs
  - Keep mentions? Airline mentions may be informative — test both variants
  - Lemmatization: optional (requires `nltk`)

### 4. Negative Reasons: `negativereason`
- Categories include: `Late Flight`, `Customer Service Issue`, `Bad Flight`, `Can't Tell`, `Flight Booking Problems`
- **Potential extensions**:
  - Multi-label classification: predict both sentiment + reason
  - Hierarchical model: first sentiment, then reason (if negative)

### 5. Temporal & Geographic Columns
- `tweet_created`: tweets from Feb–Mar 2015 (US-centric)
- `tweet_coord` / `tweet_location`: ~30–40% missing, free-text format
- **Use case**: analyze if sentiment varies by time of day or region (exploratory only)

## Data Quality Notes
- **Missing values**: Common in `negativereason`, `tweet_coord`, `tweet_location`
- **Label noise**: Low-confidence labels may be incorrect — leverage confidence columns
- **Class imbalance**: Negative class dominates — monitor F1/ROC-AUC, not just accuracy
- **Duplicates**: Check `tweet_id` for uniqueness

## Recommended Preprocessing Pipeline
```python
# 1. Load & filter
df = pd.read_csv("data/Tweets.csv")
df = df[df["airline_sentiment_confidence"] >= 0.7]  # optional

# 2. Select columns
df = df[["airline_sentiment", "text", "airline_sentiment_confidence"]].dropna()

# 3. Encode target
target_map = {"positive": 1, "negative": 0, "neutral": 2}
df["target"] = df["airline_sentiment"].map(target_map)

# 4. Preprocess text (see src/preprocessing.py)
# 5. Vectorize & train
```

## Suggested Analysis Directions
1. **Confidence vs. Accuracy**: Do high-confidence labels lead to better model performance?
2. **Airline comparison**: Which airline has the most negative/positive mentions?
3. **Reason analysis**: What are the top reasons for negative sentiment per airline?
4. **Temporal trends**: Are complaints more frequent during certain hours/days?