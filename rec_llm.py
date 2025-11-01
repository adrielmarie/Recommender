import pandas as pd
import numpy as np
import json
import time
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
        series = item['Series']
        if series == "":
            ratings_summary.append(
                f"- Rating: {item['rating']} stars\n"
                f"  Tags: {item['Character-centric']} {item['Tags']}\n"
            )
        else:
            ratings_summary.append(
                f"- Rating: {item['rating']} stars"
                f" | Tags: {item['Character-centric']} {item['Series']} {item['Tags']}\n"
            )
    ratings_text = "\n".join(ratings_summary)
    print("Ratings text: ", ratings_text)

    # Define the system prompt and query
    system_prompt = """
    You are an expert recommender system analyst. Your job is to analyze a user's item
    ratings and return a structured JSON object of their preferences.

    The user provides ratings (1-5 stars) and some attributes about the product.
    - 4-5 stars = loves
    - 1-2 stars = hates
    
    You must identify the key features (characters, types, series, tags) and
    assign weights based on the user's ratings.

    Token weights should be:
    - 6 for high-priority tokens (e.g., a specific character in a 5-star rating)
    - 3 for medium-priority tokens (e.g., a product type in a 5-star rating)
    - 2 for high-priority negative tokens (e.g., a specific character in a 1-star rating)
    - 1 for all other tokens.
      
    The input format is:
    - Rating: [1-5] stars | Tags: [tag1, tag2, tag3, ...]

    RULES:
    1.  Analyze the 'Tags' field for preferences.
    2.  "Loves" (4-5 stars) go in the "loves" array.
    3.  "Hates" (1-2 stars) go in the "hates" array.
    4.  Ignore 3-star ratings.
    5.  The JSON MUST follow this schema:
        {
          "loves": [{"token": "string", "weight": int}, ...],
          "hates": [{"token": "string", "weight": int}, ...]
        }
    6.  Do not include 3-star ratings in the output.
    7.  If there are no loves or hates, return an empty array for that key.
    """
    
    user_query = f"""
    Here are the user's ratings:

    {ratings_text}

    Return the structured JSON object.
    """

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

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
                start_time = time.time()
                response = await client.post(
                    api_url,
                    headers={'Content-Type': 'application/json'},
                    json=payload
                )
                end_time = time.time()

                duration = end_time - start_time
                
                response.raise_for_status()
                
                result = response.json()
                json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                
                if not json_text:
                    raise ValueError("API returned empty response.")
                
                print(f"LLM JSON Response:\n{json_text}")
                return json.loads(json_text), duration

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
    phrase_cols = ['Type', 'Series', 'Collaboration', 'Character-centric']

    # Handle normal phrase columns
    for col in phrase_cols:
        df[col] = df[col].fillna('').astype(str).str.lower().str.replace(r'[\s,]+', '_', regex=True)
        # Remove any trailing underscores that might result from ", "
        df[col] = df[col].str.replace(r'_+', '_', regex=True).str.strip('_')
        # Handle 'none' string from fillna/conversion
        df[col] = df[col].replace('none', '')

    # Processing columns set up as lists
    def process_list_column(tag_string):
        if not isinstance(tag_string, str):
            return ""
        
        tags = tag_string.split(',')
        processed_tags = []
        for tag in tags:
            clean_tag = tag.strip()
            if clean_tag:
                # Replace internal spaces with underscores
                tokenized_tag = clean_tag.replace(' ', '_')
                processed_tags.append(tokenized_tag)
        
        # Join processed tags with a space
        return ' '.join(processed_tags)

    # Handle tags
    list_cols = ['Characters', 'Tags']
    for col in list_cols:
        if col in df.columns:
            # Ensure it's a string, lowercase
            df[col] = df[col].fillna('').astype(str).str.lower()
            df[col] = df[col].apply(process_list_column)
            # Clean up any double underscores
            df[col] = df[col].str.replace(r'_+', '_', regex=True)

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
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'\S+')
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