import pandas as pd
import numpy as np
import json
import os
import asyncio
import httpx
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Get structured JSON of user preferences from dataframe
async def get_llm_profile(rated_items_df, api_key):
    # Format user ratings into a string for the prompt
    ratings_summary = []
    for _, item in rated_items_df.iterrows():
        ratings_summary.append(
            f"- Rating: {item['rating']} stars\n"
            f"  Title: {item['Title']}\n"
            f"  Type: {item['Type']}\n"
            f"  Main Character: {item['Character-centric']}\n"
            f"  Tags: {item['Tags']}\n"
        )
    ratings_text = "\n".join(ratings_summary)

    # Define the system prompt and query
    system_prompt = """
    You are an expert recommender system analyst. Your job is to analyze a user's item
    ratings and return a structured JSON object of their preferences.

    The user provides ratings (1-5 stars) for Sanrio products.
    - 5 stars = loves
    - 1 star = hates
    
    You must identify the key features (characters, types, series, tags) and
    assign weights based on the user's ratings.
    
    Tokenization Rules:
    1.  All tokens must be lowercase.
    2.  Replace all spaces and commas with a single underscore (e.g., 'Hello Kitty' -> 'hello_kitty'). The exception is badtz-maru, dash (-) not underscore (_).
    3.  For the 'Series' column: Tokenize the series name *exactly* as it appears (lowercased, with underscores). 
        **Do NOT append the word '_series' to the token.** (e.g., 'I Love Me' -> 'i_love_me', NOT 'i_love_me_series').
    4.  For 'Tags': Tokenize each individual tag. (e.g., "7'' plush" -> "7''_plush").

    Weighting Rules:
    - 'main_focus_character' (from 'Character-centric' column): weight 6.0
    - 'type' (from 'Type' column): weight 3.0
    - 'series' (from 'Series' column): weight 2.0
    - 'character' (from 'Characters' column): weight 1.0
    - all other 'tags': weight 1.0
      
    Return ONLY a valid JSON object in the following format:
    {
      "loves": [
        {"token": "token_name", "weight": 6.0},
        {"token": "other_token", "weight": 1.0}
      ],
      "hates": [
        {"token": "hated_token", "weight": 6.0}
      ]
    }
    """
    
    user_query = f"""
    Here are the user's ratings:
    {ratings_text}

    Analyze these ratings and provide the structured JSON output.
    Remember to tokenize keywords (e.g., 'Hello Kitty' -> 'hello_kitty', '7'' plush' -> '7''_plush').
    """

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "loves": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "token": {"type": "STRING"},
                                "weight": {"type": "NUMBER"}
                            }
                        }
                    },
                    "hates": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "token": {"type": "STRING"},
                                "weight": {"type": "NUMBER"}
                            }
                        }
                    }
                }
            }
        }
    }

    # Make the API Call
    print("Calling Gemini API...")
    
    max_retries = 3
    delay = 1.0 # Initial delay in seconds
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    api_url,
                    headers={'Content-Type': 'application/json'},
                    json=payload
                )
                
                response.raise_for_status()
                
                result = response.json()
                json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                
                if not json_text:
                    raise ValueError("API returned empty response.")
                
                print(f"LLM JSON Response:\n{json_text}")
                return json.loads(json_text)

            except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, ValueError, IndexError, KeyError) as e:
                print(f"Error calling LLM (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    print("Max retries reached. Failing.")
                    # Fallback to an empty profile
                    return {"loves": [], "hates": []}

    # Fallback in case of unexpected exit
    return {"loves": [], "hates": []}

# Doesn't have weights for items anymore
def create_tfidf_matrix(df):
    # Columns to use for content features
    text_cols = ['Characters', 'Type', 'Tags', 'Series', 'Collaboration', 'Character-centric']
    phrase_cols = ['Characters', 'Type', 'Series', 'Collaboration', 'Character-centric']

    # Handle normal phrase columns
    for col in phrase_cols:
        df[col] = df[col].fillna('').astype(str).str.lower().str.replace(r'[\s,]+', '_', regex=True)
        # Remove any trailing underscores that might result from ", "
        df[col] = df[col].str.replace(r'_+', '_', regex=True).str.strip('_')
        # Handle 'none' string from fillna/conversion
        df[col] = df[col].replace('none', '')

    # Handle tags
    col = 'Tags'
    df[col] = df[col].fillna('').astype(str).str.lower()
    df[col] = df[col].apply(lambda x:
        ' '.join([
            # Strip whitespace, replace internal spaces with _
            tag.strip().replace(' ', '_') for tag in x.split(',') if tag.strip()
        ])
    )
    # Clean up any double underscores
    df[col] = df[col].str.replace(r'_+', '_', regex=True)

    # Build the weighted content soup string by repeating the data
    df['content_soup'] = (
        df['Character-centric'] + ' ' +
        df['Type'] + ' ' +
        df['Series'] + ' ' +
        df['Characters'] + ' ' +
        df['Tags'] + ' ' +
        df['Collaboration']
    )

    # Clean up soup: remove extra spaces
    df['content_soup'] = df['content_soup'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Create a mapping from 'Item #' to its index
    item_id_to_index = pd.Series(df.index, index=df['Item #']).to_dict()

    # Compute TF-IDF feature matrix from context soup
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_soup'])

    return tfidf_matrix, tfidf, item_id_to_index

# Get recs based on profile from LLM
def get_llm_recommendations(profile_json, df, tfidf_matrix, tfidf_vectorizer, rated_item_ids, top_n=10):  
    try:
        vocab = tfidf_vectorizer.vocabulary_
        num_features = len(vocab)
        user_profile = np.zeros(num_features)

        # Failsafes for incorrect formatting
        for preference_list in [profile_json.get("loves", []), profile_json.get("hates", [])]:
            for item in preference_list:
                token = item.get("token", "")

                # Sometimes LLM gets the series wrong by adding "_series" to the end
                if token.endswith("_series"):
                    # Remove the "_series" suffix
                    corrected_token = token[:-7] 
                    # Check if the corrected token is in the vocab
                    if corrected_token in vocab:
                        item["token"] = corrected_token

                # If it gets badtz-maru wrong
                if token == "badtz_maru":
                    if "badtz-maru" in vocab:
                        item["token"] = "badtz-maru"
        
        # Add "loves" to the profile (Positive weights)
        for item in profile_json.get("loves", []):
            term = item.get("token")
            weight = item.get("weight", 1.0)
            
            if term in vocab:
                term_index = vocab[term]
                # Get pre-calculated IDF score for this term
                idf_score = tfidf_vectorizer.idf_[term_index]
                # Add the weighted score (TF * IDF)
                user_profile[term_index] += weight * idf_score
            else:
                print(f"LLM 'love' token not in vocab: {term}")

        # Subtract "hates" from the profile (Negative weights)
        for item in profile_json.get("hates", []):
            term = item.get("token")
            weight = item.get("weight", 1.0)
            
            if term in vocab:
                term_index = vocab[term]
                idf_score = tfidf_vectorizer.idf_[term_index]
                # Subtract the weighted score
                user_profile[term_index] -= weight * idf_score
            else:
                print(f"LLM 'hate' token not in vocab: {term}")
        
        # Calculate Similarity
        user_profile_sparse = csr_matrix(user_profile)
        cos_sim_scores = cosine_similarity(user_profile_sparse, tfidf_matrix).flatten()

        # Format and return results
        df_scores = pd.DataFrame({
            'Item #': df['Item #'],
            'Title': df['Title'],
            'Type': df['Type'],
            'Character-centric': df['Character-centric'],
            'Similarity': cos_sim_scores
        })

        # Filter out items that were already rated
        df_recommendations = df_scores[~df_scores['Item #'].isin(rated_item_ids)]
        
        # Filter out negative scores
        df_recommendations = df_recommendations[df_recommendations['Similarity'] > 0]
        df_recommendations = df_recommendations.sort_values(by='Similarity', ascending=False)

        return df_recommendations.head(top_n)

    except Exception as e:
        print(f"Error in get_llm_recommendations: {e}")
        return pd.DataFrame()