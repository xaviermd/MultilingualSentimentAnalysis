# Multilingual Sentiment Analysis
How good has machine learning become? Training multiple models is costly. Could one train a single Sentiment Analysis model, and use it in combination with
pre-trained Machine Translation algorithms for Multilingual Sentiment Analysis?

This project used a custom chrome extension to perform scraping of Amazon reviews on _amazon.com_ (English), _amazon.fr_ (French), _amazon.es_ (Spanish),
and _amazon.cn_ (Chinese). For each of French, Spanish, and Chinese, 22 000 reviews were translated to English using Google Translate (specifically the
GOOGLETRANSLATE() function in Google Sheets). Sentiment Analysis models were then trained on all of the datasets (English, French, French-to-English,
Spanish, etc.).

In the end, our evidence shows that Sentiment Analysis performance was higher for translated-to-English reviews in French and Spanish, but lower for Chinese.
This suggests that using Sentiment Analysis and machine translation between Indo-European languages (e.g., French, Spanish, English) may work, but that a
specific model is still required for Sino-Tibetan languages (e.g., Chinese).
