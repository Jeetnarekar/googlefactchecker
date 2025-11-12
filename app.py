
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import spacy
import sqlite3
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re
import os
import requests
from bs4 import BeautifulSoup
from collections import Counter
import math
import html
import importlib_metadata
import importlib.metadata
import requests
import streamlit as st
import pandas as pd
importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
from urllib.parse import quote_plus
import urllib

# ---------------------- Config & Models ----------------------
st.set_page_config(page_title="PolitiFact FactCheck Hub", layout="wide")


# ✅ Load API key automatically
try:
    api_key = st.secrets["google_api"]["FACT_CHECK_API_KEY"]
    st.success("✅ Google API key loaded from secrets.")
except Exception:
    api_key = "AIzaSyB6zOXxH-S_Vjef0OK3dG4a-k4ZqOa123o"   # <-- for local testing only
    st.warning("⚠️ Using key from code (not safe for deployment).")


# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load models and vectorizer (assumes files exist)
vectorizer = joblib.load("tfidf_vectorizer.joblib")
lr_model = joblib.load("logistic_regression_model.joblib")
nb_model = joblib.load("naive_bayes_model.joblib")
svc_model = joblib.load("svc_model.joblib")

import requests

def smart_query_variants(statement):
    """Generate multiple meaningful variants of the statement for broader matching."""
    doc = nlp(statement)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    entities = [ent.text for ent in doc.ents]

    variants = []
    if entities:
        variants.append(" ".join(entities))
    variants.append(" ".join(keywords))
    variants.append(statement)
    return list(set(v for v in variants if v.strip()))


def google_fact_check(statement, api_key):
    """Enhanced: broader matching using multiple query variants."""
    results = []
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    for query in smart_query_variants(statement):
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"{base_url}?query={encoded}&key={api_key}"
            r = requests.get(url, timeout=10)
            data = r.json()

            if "claims" in data and data["claims"]:
                claim = data["claims"][0]
                review = claim.get("claimReview", [{}])[0]
                return {
                    "status": "OK",
                    "query_used": query,
                    "verdict": review.get("textualRating", "Unknown"),
                    "source": review.get("publisher", {}).get("name", "Unknown"),
                    "claim_text": claim.get("text", statement),
                    "raw": data
                }

            results.append({"query": query, "found": False})

        except Exception as e:
            return {"status": "Error", "error": str(e)}

    # if none of the queries returned a match
    return {
        "status": "No results",
        "verdict": "No verified claim found (semantic search exhausted)",
        "source": "N/A",
        "claim_text": statement,
        "attempted_queries": results
    }


def spacy_preprocess(text):
    doc = nlp(str(text))
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# credentials loader: prefer st.secrets, else credentials.json
def load_google_credentials():
    cred_path = "credentials.json"
    if os.path.exists(cred_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                cred_path,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            return credentials
        except Exception as e:
            st.error(f"Error loading credentials: {e}")
            return None
    else:
        return None


# ---------------------- Google File Checker (separate window) ----------------------
def check_google_file(file_id):
    credentials = load_google_credentials()
    if credentials is None:
        return {"status": "error", "message": "No Google credentials found (st.secrets['google'] or credentials.json)."}
    try:
        service = build("drive", "v3", credentials=credentials)
        file = service.files().get(fileId=file_id, fields="name, mimeType, size, trashed").execute()
        if file.get("trashed"):
            return {"status": "error", "message": f"🚫 File '{file['name']}' is in trash"}
        if not (file["mimeType"].startswith("text/") or file["mimeType"].endswith("csv")):
            return {"status": "error", "message": f"❌ Unsupported file type: {file['mimeType']}"}
        if int(file.get("size", 0)) > 10_000_000:
            return {"status": "warning", "message": "⚠️ File is large (>10MB). Proceed with caution."}
        return {"status": "ok", "message": f"✅ File '{file['name']}' passed Google safety checks."}
    except Exception as e:
        return {"status": "error", "message": f"Google API Error: {e}"}

# ---------------------- Politifact Scraper ----------------------
# NOTE: politifact structure changes over time — this scraper is a best-effort and may need tweaks.
def scrape_politifact_claims(start_url, max_pages=1, max_claims=200):
    """
    Scrapes politifact claim pages starting from start_url (a listing or tag page).
    Returns a dataframe with columns: ['claim','speaker','ruling','date','url'].
    """
    claims = []
    page_url = start_url
    crawled = 0
    for page in range(max_pages):
        try:
            r = requests.get(page_url, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Politifact listing entries often use 'o-card' or 'm-statement' - try several selectors
            items = soup.select("article, .m-statement, .o-listicle__item") or []
            for it in items:
                if crawled >= max_claims:
                    break
                # attempt to extract link
                link_tag = it.find("a", href=True)
                href = None
                if link_tag:
                    href = link_tag["href"]
                    if href.startswith("/"):
                        href = "https://www.politifact.com" + href
                # visit claim page
                if href:
                    try:
                        rc = requests.get(href, timeout=12)
                        rc.raise_for_status()
                        csoup = BeautifulSoup(rc.text, "html.parser")
                        # claim text
                        claim_el = csoup.select_one(".m-statement__quote, .statement__text, .m-statement__quote-text, .claim")
                        claim_text = claim_el.get_text(separator=" ").strip() if claim_el else ""
                        # speaker
                        speaker_el = csoup.select_one(".m-statement__speaker, .statement__speaker, .byline a")
                        speaker = speaker_el.get_text(separator=" ").strip() if speaker_el else ""
                        # ruling (True/False/Mostly True etc)
                        ruling_el = csoup.select_one(".m-statement__meter img, .c-face, .m-claim__meter, .rating-image")
                        ruling = None
                        if ruling_el:
                            # try alt text on image or adjacent text
                            ruling = ruling_el.get("alt") or ruling_el.get_text(separator=" ").strip()
                        # some pages have a textual ruling element
                        if not ruling:
                            ruling_text_el = csoup.select_one(".m-statement__meter--text, .rating")
                            ruling = ruling_text_el.get_text(separator=" ").strip() if ruling_text_el else ""
                        # date
                        date_el = csoup.select_one("time, .m-statement__date, .published")
                        date = date_el.get("datetime") if date_el and date_el.has_attr("datetime") else (date_el.get_text(strip=True) if date_el else "")
                        claims.append({
                            "claim": html.unescape(claim_text),
                            "speaker": speaker,
                            "ruling": ruling,
                            "date": date,
                            "url": href
                        })
                        crawled += 1
                    except Exception:
                        continue
            # try to find 'next' link on listing page
            next_link = soup.select_one("a[rel='next'], .next a")
            if next_link and next_link.has_attr("href"):
                next_href = next_link["href"]
                if next_href.startswith("/"):
                    page_url = "https://www.politifact.com" + next_href
                else:
                    page_url = next_href
            else:
                break
        except Exception:
            break
    return pd.DataFrame(claims)

# ---------------------- Feature Extraction ----------------------
HEDGE_WORDS = set(["might","could","may","possibly","seems","appears","suggests","arguably","perhaps","likely","unlikely"])
MODAL_WORDS = set(["must","should","will","would","can","could","may","might","shall"])

def extract_features(text):
    doc = nlp(str(text))
    tokens = [t for t in doc if not t.is_space]
    words = [t.text for t in tokens if not t.is_punct]
    lemmas = [t.lemma_.lower() for t in tokens if not t.is_punct]
    # Lexical
    token_count = len(words)
    avg_word_len = (sum(len(w) for w in words) / token_count) if token_count else 0
    vocab_size = len(set(w.lower() for w in words))
    lemma_vocab = len(set(lemmas))
    # Syntactic
    pos_counts = Counter([t.pos_ for t in tokens])
    dep_counts = Counter([t.dep_ for t in tokens])
    # Semantic: try to get a vector length & magnitude
    try:
        vec = doc.vector
        vec_norm = float(np.linalg.norm(vec)) if vec is not None else 0.0
    except Exception:
        vec_norm = 0.0
    # Discourse/pragmatic heuristics
    sentence_count = len(list(doc.sents))
    pronouns = sum(1 for t in tokens if t.pos_ == "PRON")
    pronoun_ratio = pronouns / token_count if token_count else 0
    hedge_count = sum(1 for w in words if w.lower() in HEDGE_WORDS)
    modal_count = sum(1 for w in words if w.lower() in MODAL_WORDS)
    exclamation = text.count("!")
    question = text.count("?")
    # Simple pragmatic polarity heuristic (wordlist)
    pos_words = set(["good","right","correct","true","benefit","win","positive"])
    neg_words = set(["bad","false","wrong","lie","negative","harm"])
    pos_count = sum(1 for w in words if w.lower() in pos_words)
    neg_count = sum(1 for w in words if w.lower() in neg_words)
    polarity = (pos_count - neg_count) / (token_count+1)
    return {
        "token_count": token_count,
        "avg_word_len": avg_word_len,
        "vocab_size": vocab_size,
        "lemma_vocab": lemma_vocab,
        "pos_counts": dict(pos_counts),
        "dep_counts": dict(dep_counts),
        "semantic_vector_norm": vec_norm,
        "sentence_count": sentence_count,
        "pronoun_ratio": pronoun_ratio,
        "hedge_count": hedge_count,
        "modal_count": modal_count,
        "exclamation": exclamation,
        "question": question,
        "polarity_heuristic": polarity
    }


# 
# ---------------------- Humorous Critic ----------------------
def humorous_critic(claim_text, ruling, feat_dict=None):
    """
    Return a short humorous critique string based on ruling and simple features.
    """
    r = (ruling or "").lower()
    if "pants" in r or "pants on fire" in r:
        tone = "🔥 Pants on fire — someone set the truth on holiday."
    elif "true" in r and "mostly" not in r:
        tone = "✅ That one checks out. Give yourself a gold star (and maybe a cookie)."
    elif "mostly true" in r:
        tone = "👍 Mostly true — like a diet soda: close enough for some people."
    elif "half true" in r or "mixture" in r:
        tone = "🔀 Half-true — like ordering pizza with half the toppings you asked for."
    elif "false" in r:
        tone = "❌ False — not even close. Dramatic improvisation, perhaps?"
    else:
        tone = "🤔 The fact-checker was indecisive. Humans are complicated."
    # add flavor based on features
    if feat_dict:
        if feat_dict.get("hedge_count", 0) > 1:
            tone += " Also — lots of hedging. Sounds unsure."
        if feat_dict.get("exclamation", 0) > 0:
            tone += " And lots of ! — confident shouting does not equal truth."
        if feat_dict.get("polarity_heuristic", 0) > 0.3:
            tone += " Positive spin detected."
    return tone


def get_morphological_analysis(text):
    """
    Perform morphological and linguistic analysis using spaCy.
    Returns a list of dictionaries for easy DataFrame conversion.
    """
    if not text.strip():
        return [{"Error": "No text provided"}]
    
    doc = nlp(text)
    results = []

    for token in doc:
        results.append({
            "Token": token.text,
            "Lemma": token.lemma_,
            "Part of Speech": token.pos_,
            "Tag": token.tag_,
            "Morphology": token.morph.to_dict(),
            "Dependency": token.dep_,
            "Head Word": token.head.text
        })

    return results

# ---------------------- Benchmarking ----------------------
def benchmark_models_on_df(df, text_col, target_col):
    """
    Returns a small dict of metrics per model using SMOTE balancing.
    Expects target_col to be binary (0/1).
    """
    df = df.dropna(subset=[text_col, target_col]).copy()
    df["clean_text"] = df[text_col].astype(str).apply(spacy_preprocess)
    X = vectorizer.transform(df["clean_text"])
    y = df[target_col].astype(int).values
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        # fallback: if smote can't process sparse matrix shape, convert to dense (warning)
        X_res = X.toarray()
        X_res, y_res = smote.fit_resample(X_res, y)
    results = {}
    for name, model in [("Logistic Regression", lr_model), ("Naive Bayes", nb_model), ("SVC", svc_model)]:
        try:
            y_pred = model.predict(X_res)
            results[name] = {
                "accuracy": float(accuracy_score(y_res, y_pred)),
                "f1": float(f1_score(y_res, y_pred)),
                "precision": float(precision_score(y_res, y_pred)),
                "recall": float(recall_score(y_res, y_pred))
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# ---------------------- UI -- Main Navigation ----------------------
main_menu = st.sidebar.radio("Main", [
    "Fake vs Real Detector",
    "SQL Data Analysis / Dataset Dashboard",
    "Politifact Scraper & FactCheck",
    "Feature Analysis",
    "Benchmarking",
    "Google Fact Checker"
])

# ---------------------- 1) Fake vs Real Detector (existing) ----------------------
if main_menu == "Fake vs Real Detector":
    st.title("🧠 PolitiFact True/False Prediction (Single / CSV)")
    input_option = st.radio("Select input type:", ["Single Statement", "Upload CSV"], index=0)

    if input_option == "Single Statement":
        text_input = st.text_area("Enter a political statement:")
        sub_option = st.radio("Output Type:", ["Accuracy & F1 Comparison", "Confidence Comparison", "Morphological Analysis", "Humorous Critic"], key="single_nested")
        if st.button("Analyze / Predict"):
            if text_input.strip() == "":
                st.warning("Please enter a statement.")
            else:
                clean_text = spacy_preprocess(text_input)
                X_input = vectorizer.transform([clean_text])
                # predictions
                pred_lr = lr_model.predict(X_input)[0]
                prob_lr = lr_model.predict_proba(X_input)[0][1] if hasattr(lr_model, "predict_proba") else 0
                pred_nb = nb_model.predict(X_input)[0]
                prob_nb = nb_model.predict_proba(X_input)[0][1] if hasattr(nb_model, "predict_proba") else 0
                pred_svc = svc_model.predict(X_input)[0]
                prob_svc = 0.9 if pred_svc == 1 else 0.1
                st.subheader("🧩 Model Predictions")
                st.metric("Logistic Regression", "True" if pred_lr == 1 else "False", f"{prob_lr*100:.1f}%")
                st.metric("Naive Bayes", "True" if pred_nb == 1 else "False", f"{prob_nb*100:.1f}%")
                st.metric("SVC", "True" if pred_svc == 1 else "False", f"{prob_svc*100:.1f}%")

                if sub_option == "Morphological Analysis":
                    st.subheader("🔤 Morphological & Linguistic Features")
                    st.dataframe(pd.DataFrame(get_morphological_analysis(text_input)))

                elif sub_option == "Accuracy & F1 Comparison":
                    st.subheader("📊 Accuracy & F1 (SMOTE Balanced) — Full Dataset")
                    if os.path.exists("politifact_dataset.csv"):
                        df_full = pd.read_csv("politifact_dataset.csv").dropna(subset=["statement", "BinaryNumTarget"])
                        df_full["clean_statement"] = df_full["statement"].apply(spacy_preprocess)
                        X = vectorizer.transform(df_full["clean_statement"])
                        y = df_full["BinaryNumTarget"].astype(int)
                        smote = SMOTE(random_state=42)
                        X_res, y_res = smote.fit_resample(X, y)
                        y_pred_lr = lr_model.predict(X_res)
                        y_pred_nb = nb_model.predict(X_res)
                        y_pred_svc = svc_model.predict(X_res)
                        accuracies = {
                            "Logistic Regression": accuracy_score(y_res, y_pred_lr),
                            "Naive Bayes": accuracy_score(y_res, y_pred_nb),
                            "SVC": accuracy_score(y_res, y_pred_svc)
                        }
                        f1_scores = {
                            "Logistic Regression": f1_score(y_res, y_pred_lr),
                            "Naive Bayes": f1_score(y_res, y_pred_nb),
                            "SVC": f1_score(y_res, y_pred_svc)
                        }
                        st.bar_chart(pd.DataFrame({"Accuracy": accuracies, "F1-Score": f1_scores}))
                    else:
                        st.warning("politifact_dataset.csv not found in project directory.")

                elif sub_option == "Confidence Comparison":
                    st.subheader("📊 Model Confidence for this Statement")
                    conf_scores = {"Logistic Regression": prob_lr, "Naive Bayes": prob_nb, "SVC": prob_svc}
                    st.bar_chart(pd.DataFrame.from_dict(conf_scores, orient="index", columns=["confidence"]))

                elif sub_option == "Humorous Critic":
                    feats = extract_features(text_input)
                    critique = humorous_critic(text_input, "", feat_dict=feats)
                    st.subheader("🎭 Humorous Critic")
                    st.write(critique)

    else:
        # CSV upload mode
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        file_url = st.text_input("If from Google Drive, paste its share link (optional):")
        # optional Google file check
        if file_url:
            match = re.search(r"/d/([a-zA-Z0-9_-]+)", file_url)
            if match:
                file_id = match.group(1)
                result = check_google_file(file_id)
                st.info(result["message"])
                if result["status"] != "ok":
                    st.stop()
            else:
                st.warning("Invalid Google Drive URL format.")
        if uploaded_file is not None:
            df_csv = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully.")
            st.dataframe(df_csv.head(5))
            # compute confidences and optional target
            text_col = st.selectbox("Select text column:", df_csv.columns)
            target_col = st.selectbox("Select target column (optional):", [None] + list(df_csv.columns))
            df_csv = df_csv.dropna(subset=[text_col])
            df_csv["clean_statement"] = df_csv[text_col].apply(spacy_preprocess)
            df_csv["LR_Prob"] = lr_model.predict_proba(vectorizer.transform(df_csv["clean_statement"]))[:, 1]
            df_csv["NB_Prob"] = nb_model.predict_proba(vectorizer.transform(df_csv["clean_statement"]))[:, 1]
            df_csv["SVC_Pred"] = svc_model.predict(vectorizer.transform(df_csv["clean_statement"]))
            df_csv["SVC_Prob"] = np.where(df_csv["SVC_Pred"] == 1, 0.9, 0.1)
            st.dataframe(df_csv.head(10))
            if target_col:
                st.subheader("📊 Accuracy & F1 (SMOTE Balanced) on Uploaded CSV")
                results = benchmark_models_on_df(df_csv, text_col, target_col)
                st.json(results)

# ---------------------- 2) SQL Data Analysis / Dataset Dashboard ----------------------
elif main_menu == "SQL Data Analysis / Dataset Dashboard":
    st.title("📊 PolitiFact Dataset Dashboard / SQL Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    file_url = st.text_input("If from Google Drive, paste its share link (optional):")
    if file_url:
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", file_url)
        if match:
            file_id = match.group(1)
            result = check_google_file(file_id)
            st.info(result["message"])
            if result["status"] != "ok":
                st.stop()
        else:
            st.warning("Invalid Google Drive URL format.")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select text column:", df.columns)
        target_col = st.selectbox("Select target column:", [None] + list(df.columns))
        speaker_col = st.selectbox("Select speaker column:", df.columns)
        df[text_col] = df[text_col].astype(str)
        df["clean_text"] = df[text_col].apply(spacy_preprocess)
        conn = sqlite3.connect(":memory:")
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        query_option = st.selectbox("Select Query:", [
            "Count Total Unique Speakers",
            "List All Unique Speakers",
            "Top 5 Speakers by Number of Statements",
            "Count True and False Statements per Speaker",
            "Count Statements by Rating",
            "Speakers with Only False Statements",
            "Top Speakers by True Statements"
        ])
        queries = {
            "Count Total Unique Speakers": f"SELECT COUNT(DISTINCT {speaker_col}) as total_unique_speakers FROM dataset;",
            "List All Unique Speakers": f"SELECT DISTINCT {speaker_col} as unique_speaker FROM dataset;",
            "Top 5 Speakers by Number of Statements": f"SELECT {speaker_col}, COUNT(*) as statement_count FROM dataset GROUP BY {speaker_col} ORDER BY statement_count DESC LIMIT 5;",
            "Count True and False Statements per Speaker": f"SELECT {speaker_col}, SUM({target_col}) as true_count, COUNT(*) - SUM({target_col}) as false_count FROM dataset GROUP BY {speaker_col};",
            "Count Statements by Rating": f"SELECT {target_col} as rating, COUNT(*) as count FROM dataset GROUP BY {target_col};",
            "Speakers with Only False Statements": f"SELECT {speaker_col} FROM dataset GROUP BY {speaker_col} HAVING SUM({target_col})=0;",
            "Top Speakers by True Statements": f"SELECT {speaker_col}, SUM({target_col}) as true_count FROM dataset GROUP BY {speaker_col} ORDER BY true_count DESC LIMIT 5;"
        }
        result = pd.read_sql_query(queries[query_option], conn)
        st.subheader(f"Result: {query_option}")
        st.dataframe(result)
        numeric_cols = [c for c in result.columns if "count" in c or "true_count" in c or "false_count" in c]
        if numeric_cols:
            st.bar_chart(result.set_index(result.columns[0])[numeric_cols[0]])
        if target_col is not None:
            st.subheader("📊 ML Metrics on Uploaded Dataset (SMOTE Balanced)")
            bm = benchmark_models_on_df(df, "clean_text", target_col)
            st.json(bm)

# ---------------------- 3) Politifact Scraper & FactCheck ----------------------
elif main_menu == "Politifact Scraper & FactCheck":
    st.title("🕵️ Politifact Scraper & Quick FactCheck")
    st.markdown("Provide a Politifact listing URL (e.g., category/tag or homepage) and press *Scrape*. The scraper will visit claim pages and collect claim, speaker, ruling, date, url.")
    start_url = st.text_input("Start URL (Politifact listing page):", value="https://www.politifact.com/factchecks/")
    pages = st.number_input("Max listing pages to crawl:", min_value=1, max_value=10, value=1)
    max_claims = st.number_input("Max claims to fetch:", min_value=1, max_value=500, value=50)
    if st.button("Scrape Politifact"):
        with st.spinner("Scraping... this may take a while depending on site and network"):
            df_claims = scrape_politifact_claims(start_url.strip(), max_pages=pages, max_claims=max_claims)
        if df_claims.empty:
            st.warning("No claims scraped — site structure may differ or scraping blocked. Try another listing URL or reduce pages.")
        else:
            st.success(f"Scraped {len(df_claims)} claims.")
            st.dataframe(df_claims.head(50))
            # store scraped dataset in session state for later
            st.session_state["scraped_claims"] = df_claims

    # Quick FactCheck: choose a scraped claim or paste one
    st.markdown("---")
    st.subheader("Quick FactCheck (Humorous + Features)")
    claim_choice = None
    if "scraped_claims" in st.session_state:
        df_claims = st.session_state["scraped_claims"]
        claim_choice = st.selectbox("Choose a scraped claim (or paste your own below)", options=["(paste your own)"] + df_claims["claim"].tolist())
    else:
        claim_choice = "(paste your own)"
    claim_text_input = st.text_area("Claim text:", value= "" if claim_choice == "(paste your own)" else claim_choice, height=120)
    ruling_input = st.text_input("If you know the ruling (optional):")
    if st.button("FactCheck this Claim"):
        if not claim_text_input.strip():
            st.warning("Enter a claim to fact-check.")
        else:
            feats = extract_features(claim_text_input)
            critique = humorous_critic(claim_text_input, ruling_input, feats)
            st.subheader("Feature Summary")
            st.json(feats)
            st.subheader("Humorous Critic")
            st.write(critique)
            st.subheader("Model Predictions (ensemble of existing models)")
            clean = spacy_preprocess(claim_text_input)
            Xtmp = vectorizer.transform([clean])
            p_lr = lr_model.predict(Xtmp)[0]
            p_nb = nb_model.predict(Xtmp)[0]
            p_svc = svc_model.predict(Xtmp)[0]
            st.write(f"Logistic Regression: {'True' if p_lr==1 else 'False'}")
            st.write(f"Naive Bayes: {'True' if p_nb==1 else 'False'}")
            st.write(f"SVC: {'True' if p_svc==1 else 'False'}")

# ---------------------- 4) Feature Analysis ----------------------
elif main_menu == "Feature Analysis":
    st.title("🔬 Feature Extraction & Analysis")
    text_sample = st.text_area("Enter text to analyze (one paragraph):", value="The government will raise taxes next year, experts say. Could be a big change!")
    if st.button("Extract Features"):
        feats = extract_features(text_sample)
        st.subheader("Lexical / Basic Stats")
        st.write({
            "Token count": feats["token_count"],
            "Avg word length": round(feats["avg_word_len"], 3),
            "Vocab size": feats["vocab_size"],
            "Lemma vocab size": feats["lemma_vocab"],
            "Sentence count": feats["sentence_count"]
        })
        st.subheader("Syntactic Distributions (top POS)")
        st.write(pd.Series(feats["pos_counts"]).sort_values(ascending=False).head(10).to_dict())
        st.subheader("Dependency Counts (top)")
        st.write(pd.Series(feats["dep_counts"]).sort_values(ascending=False).head(10).to_dict())
        st.subheader("Pragmatic / Discourse Heuristics")
        st.write({
            "Pronoun ratio": round(feats["pronoun_ratio"], 3),
            "Hedge words": feats["hedge_count"],
            "Modal verbs": feats["modal_count"],
            "Exclamation marks": feats["exclamation"],
            "Question marks": feats["question"],
            "Polarity heuristic": round(feats["polarity_heuristic"], 3)
        })
        st.subheader("Semantic Vector Norm (proxy for semantic content magnitude)")
        st.write(round(feats["semantic_vector_norm"], 4))

# ---------------------- 5) Benchmarking ----------------------
elif main_menu == "Benchmarking":
    st.title("⚖️ Benchmarking Models")
    st.markdown("Provide a CSV (with text and binary label columns) to benchmark the three models with SMOTE balancing.")
    uploaded_file = st.file_uploader("Upload CSV for benchmarking", type=["csv"])
    if uploaded_file is not None:
        df_bench = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select text column:", df_bench.columns)
        target_col = st.selectbox("Select binary target column:", df_bench.columns)
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                results = benchmark_models_on_df(df_bench, text_col, target_col)
            st.json(results)

# ---------------------- 6) Google File Checker (separate window) ----------------------
# ---------------------- 6) Google File Checker + File Uploader ----------------------
# ---------------------- 6) Google File Checker (Manual Upload) ----------------------
# ---------------------- 6) Google Fact Checker (API) ----------------------
# ---------------------- 6) Google Fact Checker (API) ----------------------
elif main_menu == "Google Fact Checker":
    st.title("🔍 Google Fact Checker (API)")
    st.markdown("Use the Google Fact Check Tools API to verify statements and compare them with your ML model predictions.")

    api_key = st.text_input("🔑 Enter your Google Fact Check API key:", type="password")

    input_mode = st.radio("Select Input Type:", ["Single Statement", "Upload CSV"])

    # ---------------------- SINGLE STATEMENT ----------------------
    if input_mode == "Single Statement":
        statement = st.text_area("🗣️ Enter a statement to verify:")

        if st.button("Check Statement"):
            if not api_key.strip():
                st.error("Please enter your Google Fact Check API key.")
            elif not statement.strip():
                st.warning("Please enter a statement.")
            else:
                # 1️⃣ Google API Fact Check
                fact = google_fact_check(statement, api_key)

                # 2️⃣ Model Predictions
                clean_text = spacy_preprocess(statement)
                X_input = vectorizer.transform([clean_text])
                pred_lr = lr_model.predict(X_input)[0]
                pred_nb = nb_model.predict(X_input)[0]
                pred_svc = svc_model.predict(X_input)[0]

                st.subheader("🧠 Model Predictions")
                st.metric("Logistic Regression", "True" if pred_lr == 1 else "False")
                st.metric("Naive Bayes", "True" if pred_nb == 1 else "False")
                st.metric("SVC", "True" if pred_svc == 1 else "False")

                st.subheader("🌐 Google Fact Check Result")
                if fact["status"] == "OK":
                    st.success(f"✅ Verdict: **{fact['verdict']}** (Source: {fact['source']})")
                    st.write(f"📰 Claim: {fact['claim_text']}")
                elif fact["status"] == "No results":
                    st.info("No fact-check results found for this statement.")
                else:
                    st.error(f"Error: {fact.get('error', 'Unknown error')}")

    # ---------------------- CSV UPLOAD ----------------------
    elif input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("📁 Upload a CSV file containing statements", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully.")
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Select the column containing statements
            text_column = st.selectbox("Select the column containing statements:", df.columns)

            if not api_key.strip():
                st.error("Please enter a valid Google Fact Check API key.")
            else:
                if st.button("🔍 Run Fact Check on Uploaded Data"):
                    results = []
                    with st.spinner("Contacting Google Fact Check API..."):
                        for i, s in enumerate(df[text_column].dropna().head(20)):  # limit to 20 for demo
                            fact = google_fact_check(s, api_key)
                            clean_text = spacy_preprocess(s)
                            X_input = vectorizer.transform([clean_text])
                            pred_lr = lr_model.predict(X_input)[0]

                            results.append({
                                "Statement": s,
                                "Model Prediction": "True" if pred_lr == 1 else "False",
                                "Google Verdict": fact["verdict"],
                                "Fact Source": fact["source"]
                            })

                    result_df = pd.DataFrame(results)
                    st.subheader("📊 Comparison Results")
                    st.dataframe(result_df)

                    # Visualization of agreement
                    st.subheader("📈 Agreement Summary")
                    same = result_df[result_df["Model Prediction"].str.lower() == result_df["Google Verdict"].str.lower()]
                    agree = len(same)
                    total = len(result_df)
                    st.write(f"✅ Agreement: **{agree}/{total} ({(agree/total)*100:.1f}% match)**")

                    st.bar_chart({
                        "Agreed": [agree],
                        "Disagreed": [total - agree]
                    })
        else:
            st.info("⬆️ Please upload a CSV file to begin.")
# 6) Google Fact Checker (API) ----------------------


# ---------------------- END ----------------------
st.markdown("---")
st.write("PolitiFact FactCheck Hub — Basic scraping, feature extraction, benchmarking, and Google Drive checking. Modify selectors or heuristics as needed. 🚀")
