# AI/ML Guide for the MIND News Recommender

## The main idea

This project is a **content-based recommender system**. It learns a numerical representation of each news title and recommends unseen articles whose representations are most similar to the articles a user has clicked.

The model is intentionally a strong baseline rather than a deep neural network. It is fast, explainable, easy to evaluate, and appropriate for demonstrating core recommendation-system engineering.

## The dataset

**MIND** means Microsoft News Dataset. Each `news.tsv` row describes an article. Each `behaviors.tsv` row describes an anonymous user's history and an impression event.

An **impression** is a group of articles shown to a user. In an impression token such as `N123-1`, `N123` is the article ID and `1` means the article was clicked. A token ending in `-0` was shown but not clicked.

The project uses `MINDlarge_train` for fitting the title representation and `MINDlarge_dev` for held-out evaluation. The test set is kept separate for a final experiment and is not used to tune the model.

## Core machine-learning terms

### Feature

A **feature** is an input signal used by a model. In this project, the primary feature is the text of a news title. Category, subcategory, abstract, and entity information are possible future features.

### Preprocessing

**Preprocessing** converts raw data into a consistent form before modeling. The notebook removes empty titles, lowercases text, and replaces punctuation with spaces. This reduces irrelevant variation while keeping the pipeline understandable.

### Tokengit push origin main --force


A **token** is a unit of text. With the default word analyzer, a token is usually a word. The vectorizer also creates two-word phrases because the model uses unigrams and bigrams.

### Vocabulary

The **vocabulary** is the set of tokens retained by the vectorizer. Very rare terms are removed with `min_df`, and extremely common terms are limited with `max_df`. This controls memory usage and noise.

### TF-IDF

**TF-IDF** means term frequency-inverse document frequency. It gives a word a high weight when it is important in one title but not common across all titles.

For term `t` in document `d`, the intuition is:

```text
TF-IDF(t, d) = term frequency in d × inverse document frequency of t
```

This is useful for news titles because distinctive terms such as a company name, event, or topic should contribute more than words appearing in nearly every title.

### Vector space model

A **vector space model** represents each title as a numerical vector. Two titles can then be compared using mathematical operations instead of direct string matching.

The trained item matrix has one row per article and one column per vocabulary term:

```text
number of articles × vocabulary size
```

The matrix is sparse because each title uses only a small fraction of the complete vocabulary.

### Sparse matrix

A **sparse matrix** stores mostly non-zero values efficiently instead of allocating memory for every zero. TF-IDF title matrices are naturally sparse. The engine keeps this representation during inference so recommendations remain practical as the number of articles grows.

### User profile

A **user profile** is the aggregate representation of a user's history. This implementation sums the TF-IDF vectors of the articles in the click history. The result is a compact representation of the user's observed interests.

### Cosine similarity

**Cosine similarity** measures the angle between two vectors. It is high when the vectors point in a similar direction and low when their term patterns differ.

```text
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

The engine computes the similarity between the user profile and every article vector, then ranks articles by score.

### Content-based filtering

**Content-based filtering** recommends items that resemble items the user already liked. It does not need other users' behavior to calculate a recommendation. Its main strength is explainability; its main limitation is that it can repeatedly recommend topics similar to the user's existing history.

### Cold start

The **cold-start problem** occurs when there is not enough history for a new user. This project uses a deterministic fallback: it returns the first available articles from the trained catalog. A future production version could replace this with a popularity-ranked list.

### Seen-item filtering

The engine removes articles already present in the user's history. This prevents the system from recommending the same clicked item again and makes evaluation more meaningful.

## Evaluation terms

### Train, development, and test sets

The **training set** is used to fit model parameters such as the vocabulary and IDF weights. The **development set**, also called validation data, is used to measure quality while making design decisions. The **test set** should be reserved for a final unbiased report.

### Top-K recommendation

A **top-K list** contains the first K ranked recommendations. For example, Recall@10 evaluates the first ten recommendations rather than the full catalog.

### Precision@K

**Precision@K** is the fraction of the top K recommendations that were clicked:

```text
relevant recommendations in top K / K
```

It answers: “How much of the returned list was relevant?”

### Recall@K

**Recall@K** is the fraction of the user's clicked target articles that appear in the top K list:

```text
relevant recommendations in top K / relevant target articles
```

It answers: “How much of the relevant content did the system recover?”

### Hit Rate@K

**Hit Rate@K** is one when at least one relevant article appears in the top K list and zero otherwise. Averaging it across examples gives the proportion of users for whom the system produced at least one hit.

### MRR@K

**Mean Reciprocal Rank**, or **MRR@K**, rewards placing the first relevant article near the top. If the first hit is at rank 1, its reciprocal rank is 1. If it is at rank 5, its reciprocal rank is 0.2. No hit contributes zero.

### Popularity baseline

A **baseline** is a simple reference system. This project counts clicked articles in the training impressions and ranks the most-clicked articles. The TF-IDF model is useful only if it provides a meaningful comparison with this simple popularity strategy.

### Offline evaluation limitation

MIND evaluation observes clicks only among articles that were shown in an impression. A recommendation can be semantically useful but still receive no credit if it was not one of the logged candidates. Therefore, offline metrics are evidence of ranking quality, not a complete measure of real user satisfaction.

## Production terms

### Inference

**Inference** is using a trained artifact to produce predictions for new input. In this project, inference means converting click history into ranked news IDs.

### Training artifact

A **training artifact** is the serialized object required for inference. `model.pkl` contains the vectorizer, sparse item matrix, news metadata, ID mapping, and experiment metadata. It is generated offline and loaded by FastAPI at startup.

### Online inference versus retraining

**Online inference** happens for every API request and should be fast. **Retraining** rebuilds the vocabulary and item matrix from data and is performed offline. A new click does not retrain this model; it changes the user profile used for the next inference request.

### Latency

**Latency** is the time between receiving a request and returning a response. The service measures it with a high-resolution performance counter and stores the result in PostgreSQL.

### Non-blocking inference

FastAPI uses an asynchronous event loop. Heavy numerical work must not freeze that loop. The engine uses `asyncio.to_thread` to execute sparse matrix calculations in a worker thread while the event loop remains available for other requests.

### Telemetry

**Telemetry** is operational data about system behavior. This project records the user ID, returned news IDs, latency, and creation time in PostgreSQL. It is useful for monitoring and debugging; it is not the model itself.

## Final implementation workflow

1. Place the raw MINDlarge folders under `data/raw/`.
2. Run the notebook with the project virtual environment selected as the kernel.
3. Review the model-versus-baseline metric table.
4. Confirm that `data/artifacts/model.pkl` was created.
5. Start FastAPI and verify `/api/v1/ready`.
6. Start Streamlit and submit valid MIND news IDs.
7. Inspect PostgreSQL telemetry after an inference request.
8. Reserve `MINDlarge_test` for a final report after the design is frozen.

## References

[1]: https://msnews.github.io/ "MIND: Microsoft News Dataset"
[2]: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction "Scikit-learn text feature extraction"
[3]: https://scikit-learn.org/stable/modules/metrics.html#pairwise-metrics-affinities-and-kernels "Scikit-learn pairwise metrics"
[4]: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval) "Information retrieval evaluation measures"
