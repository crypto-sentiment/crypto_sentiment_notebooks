# Cryptonews Sentiment Jupyter Notebooks

[Project description](https://www.notion.so/42e0593fcf9d48a78f503c7a3d0bc619) (Russian only.).

Fixed cross-validation folds for experiments: `data/folds.csv`, [notebook](notebooks/20220411_btc_4500_titles_fix_folds_for_validations.ipynb). The baseline tf-idf & logreg model hits 72.1% accuracy in this setup. 

Learning curves built for the baseline model: [task](https://www.notion.so/a74951e4e815480584dea7d61ddce6cc?v=dbfdb1207d0e451b827d3c5041ed0cfd&p=41d93e8bcd0a47949d0ed92f0f6592eb), [notebook](notebooks/20220408_btc_4500_titles_logit_tfidf_learning_curves.ipynb). Adding new labeled data helps.

<img src="figures/20220408_learning_curves_baseline_model_4500_titles.png" width=70% />

[This notebook](notebooks/20220413_scrape_bitcointicker_news_prepare_batches_for_annotation.ipynb) scrapes [https://bitcointicker.co/news](https://bitcointicker.co/news/) (Selenium + BeautifulSoup) and examines prediction entropy which is a way to distinguish easy and hard examples. 

[This notebook](notebooks/20220415_scrape_bitcointicker_news_perform_eda.ipynb) scrapes some more Bitcointicker news (27k) and performs basic EDA with the following outcomes:

- ~27k news titles are saved locally and to the `cryptonews.bitcointicker_news` Mongo collection
- Filtering based on presence of at least one verb seems to make sense
- Most of the titles under 4 words in length make no sense. Starting from 4 words is reasonable
- Placeholder titles need to be removed. E.g. "Bitcoin daily chart alert" is seen >90 times. 
