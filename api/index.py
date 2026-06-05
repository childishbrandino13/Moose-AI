from flask import Flask, request, jsonify
import requests
import os
import re
import time
from datetime import datetime
from upstash_vector import Index # Required: pip install upstash-vector

app = Flask(__name__)

# ── Config from Environment Variables ──
SLACK_BOT_TOKEN     = os.environ.get('SLACK_BOT_TOKEN')
GEMINI_API_KEY      = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL        = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
SLACK_ALERT_CHANNEL = os.environ.get('SLACK_ALERT_CHANNEL', 'C0B8581VDU1')
CACHE_SECRET        = os.environ.get('CACHE_SECRET')
ALERT_THRESHOLD     = int(os.environ.get('ALERT_THRESHOLD', '3'))

# Upstash Vector Dashboard Marketplace Configuration
VECTOR_URL          = os.environ.get('UPSTASH_VECTOR_REST_URL')
VECTOR_TOKEN        = os.environ.get('UPSTASH_VECTOR_REST_TOKEN')
vector_index        = Index(url=VECTOR_URL, token=VECTOR_TOKEN)

# ── Safety Guardrails State ──
RATE_LIMIT_WINDOW = 3600             # 1 hour tracking frame
MAX_GEMINI_CALLS_PER_HOUR = 100      # Free-tier enforcement threshold
ALERT_COOLDOWN_WINDOW = 900          # 15 minutes of quiet time between Slack pings

gemini_call_timestamps = []
last_alert_time = 0

# ── System Safety Logic ──
def check_gemini_rate_limit():
    """Safety Guardrail: Drops processing if rapid looping risks billing."""
    global gemini_call_timestamps
    now = time.time()
    
    # Prune elements older than 1 hour
    gemini_call_timestamps = [t for t in gemini_call_timestamps if now - t < RATE_LIMIT_WINDOW]
    
    if len(gemini_call_timestamps) >= MAX_GEMINI_CALLS_PER_HOUR:
        print("⚠️ SAFETY GUARDRAIL: Gemini API limit hit. Pausing processing to prevent charge.")
        return False
        
    gemini_call_timestamps.append(now)
    return True

# ── Google Gemini Embeddings Call ──
def get_embedding(text, is_query=False):
    """
    Generates numerical vectors using the active gemini-embedding-001 model.
    Correctly formats MRL parameter constraints to clip the size down to 768.
    """
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent'
    
    # Set the correct task type based on workflow context
    task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    
    body = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        # THIS IS THE FIXED PAYLOAD FORMAT FOR MODERN GEMINI MODELS:
        "outputDimensionality": 768
    }
    
    res = requests.post(url, json=body, headers={'X-goog-api-key': GEMINI_API_KEY}, timeout=15)
    res_json = res.json()
    
    if 'error' in res_json:
        raise Exception(f"Gemini Embedding Error: {res_json['error'].get('message')}")
        
    return res_json['embedding']['values']

# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Moose is alive 🐾'})

@app.route('/', methods=['POST'])
def slack_handler():
    """Handles conversational workspace user mentions with immediate loading state feedback."""
    payload = request.get_json(force=True, silent=True) or {}

    if payload.get('type') == 'url_verification':
        return jsonify({'challenge': payload.get('challenge')})

    # CRITICAL: Bypasses Slack retry spam instantly while Moose is running the Gemini process
    if request.headers.get('X-Slack-Retry-Num'):
        return jsonify({'ok': True})

    event = payload.get('event', {})
    if event.get('type') != 'app_mention' or event.get('bot_id'):
        return jsonify({'ok': True})

    text = event.get('text', '')
    if not text:
        return jsonify({'ok': True})

    question  = re.sub(r'<@[A-Z0-9]+>\s*', '', text).strip()
    channel   = event.get('channel')
    thread_ts = event.get('thread_ts') or event.get('ts')

    # STEP 1: Post an immediate placeholder message to let the team know Moose is digging
    # This responds to Slack's webhook within milliseconds, completely bypassing the 3-second timeout rule
    loading_res = requests.post(
        'https://slack.com/api/chat.postMessage',
        headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
        json={
            'channel': channel,
            'thread_ts': thread_ts,
            'text': ':cat-jam-party: _Moose is scanning database patterns..._'
        },
        timeout=10
    ).json()

    # Capture the unique timestamp of the placeholder message so we can overwrite it later
    loading_message_ts = loading_res.get('message', {}).get('ts')

    # STEP 2: Safety Guardrail evaluation
    if not check_gemini_rate_limit():
        if loading_message_ts:
            update_slack_message(channel, loading_message_ts, "⚠️ System rate limiter active. Ask again in an hour.")
        return jsonify({'ok': True})

    try:
        # STEP 3: Handle the dual namespace vector database query and get context data
        question_vector = get_embedding(question, is_query=True)
        combined_results = []
        
        # Pull matching context from Google Sheets namespace
        try:
            sheet_results = vector_index.query(
                vector=question_vector, 
                top_k=15, 
                include_metadata=True,
                namespace="google-sheets"
            )
            if sheet_results: # Ensure it's a valid list and not None
                combined_results.extend(sheet_results)
                
        except Exception as e:
            print(f"Sheets namespace query fallback: {e}")

        # Pull matching context from Helpshift namespace
        try:
            helpshift_results = vector_index.query(
                vector=question_vector, 
                top_k=15, 
                include_metadata=True,
                namespace="helpshift"
            )
            if helpshift_results: # Ensure it's a valid list and not None
                combined_results.extend(helpshift_results)
                
        except Exception as e:
            print(f"Helpshift namespace query fallback: {e}")

        # Sort globally by vector proximity score if we actually found results
        if combined_results:
            combined_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            final_matches = combined_results[:25]
        else:
            final_matches = []

        comments = []
        for res in final_matches:
            # Check if metadata exists on the result object
            meta = getattr(res, 'metadata', None)
            if meta:
                comments.append({
                    'date': meta.get('date', 'Unknown Date'),
                    'playerCode': meta.get('playerCode', 'N/A'),
                    'comment': meta.get('comment', '')
                })

        # STEP 4: Call Gemini 2.5 Flash to synthesize the final answer
        answer = answer_question(question, comments)

        # STEP 5: Overwrite the placeholder emoji message with the final response text
        if loading_message_ts:
            update_slack_message(channel, loading_message_ts, answer)
        else:
            # Fallback if the original post tracking initialization somehow dropped
            post_slack_reply(channel, thread_ts, answer)

    except Exception as e:
        print(f"Error handling query chain processing: {str(e)}")
        if loading_message_ts:
            update_slack_message(channel, loading_message_ts, f"⚠️ Sorry, I encountered an internal processing error: {str(e)}")

    return jsonify({'ok': True})

@app.route('/cache', methods=['POST'])
def cache_handler():
    """Handles manual/webhook ingestion of comments into specific vector namespaces."""
    incoming_secret = request.headers.get('X-Cache-Secret')
    if not incoming_secret or incoming_secret != CACHE_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(force=True, silent=True) or {}
    comment = data.get('comment')
    player_code = data.get('playerCode')
    date_str = data.get('date')
    
    # Extract incoming timestamp and tags array, providing robust defaults
    timestamp = data.get('timestamp', int(time.time()))
    tags = data.get('tags', ["spreadsheet"])

    if not comment:
        return jsonify({'error': 'Missing comment parameter'}), 400

    # Dynamically detect which namespace to drop this record into. Defaults to 'google-sheets'
    target_namespace = request.args.get('namespace', 'google-sheets')

    try:
        # Generate the numerical semantic vector map
        comment_vector = get_embedding(comment, is_query=False)
        
        # Formulate a unique vector ID footprint
        record_id = f"row_{player_code}_{int(time.time())}"

        # Push securely to Upstash using the targeted isolated namespace parameter
        vector_index.upsert(
            vectors=[
                (
                    record_id, 
                    comment_vector, 
                    {
                        "comment": comment, 
                        "playerCode": player_code, 
                        "date": date_str,
                        "timestamp": timestamp, # For future chronological boundaries
                        "tags": tags            # For exact frequency filtering calculations
                    }
                )
            ],
            namespace=target_namespace 
        )

        return jsonify({'status': 'success', 'id': record_id, 'namespace': target_namespace}), 200

    except Exception as e:
        print(f"Ingestion Failure Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ── Synthesis & Output Format Engines ─────────────────────────

def answer_question(question, comments):
    if comments:
        data_block = '\n'.join(
            f"{i+1}. [{c['date']}] [Support Code: {c['playerCode']}] {c['comment']}"
            for i, c in enumerate(comments)
        )
    else:
        data_block = '(No matching context references found across database profiles)'

    prompt = f"""You are a CS intelligence assistant for a mobile sweeps app game team. Answer questions about negative player feedback directly and thoroughly.

Relevant context vectors extracted from database matching:
{data_block}

Today: {datetime.now().strftime('%b %d, %Y')}
Question: "{question}"

Format rules:
- Start your answer immediately — no greetings or preambles
- Use * for bold, _ for italic.
- Bullet points must start with •
- Limit output context completely to fit safely inside 800 characters max."""

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
    
    body = {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': 1024},
        # Safety Settings: Prevents Gemini from shutting down when analyzing angry or flagged review text
        'safetySettings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    res = requests.post(url, json=body, headers={'X-goog-api-key': GEMINI_API_KEY}, timeout=45)
    res_json = res.json()

    if 'error' in res_json:
        return f"⚠️ Gemini API Error: {res_json['error'].get('message', 'Unknown error')}"
        
    if 'candidates' not in res_json or not res_json['candidates']:
        # If blocked by safety despite settings, or prompt was malformed
        prompt_feedback = res_json.get('promptFeedback', {})
        return f"⚠️ Moose couldn't parse an answer. Prompt feedback: {prompt_feedback.get('blockReason', 'Unknown block reason')}"

    return res_json['candidates'][0]['content']['parts'][0]['text']

def trigger_slack_alert(new_comment, count, player_code):
    message = (
        f"⚠️ *Moose Alert* — A negative feedback pattern has hit the threshold!\n"
        f"• *Current Volume:* {count} matching recent player issues identified.\n"
        f"• *Latest Ticket Highlight:* _\"{new_comment}\"_\n"
        f"• *Impacted Support Code Reference:* `{player_code}`\n\n"
        f"_Ask me details inside a thread mention to see historical parallels._"
    )
    requests.post(
        'https://slack.com/api/chat.postMessage',
        headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
        json={'channel': SLACK_ALERT_CHANNEL, 'text': message},
        timeout=10
    )

def post_slack_reply(channel, thread_ts, text):
    requests.post(
        'https://slack.com/api/chat.postMessage',
        headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
        json={'channel': channel, 'thread_ts': thread_ts, 'text': text},
        timeout=10
    )
    
def update_slack_message(channel, message_ts, text):
    """
    Overwrites an existing Slack message text using its transaction timestamp id.
    """
    requests.post(
        'https://slack.com/api/chat.update',
        headers={
            'Authorization': f'Bearer {SLACK_BOT_TOKEN}',
            'Content-Type': 'application/json; charset=utf-8'
        },
        json={
            'channel': channel,
            'ts': message_ts,
            'text': text
        },
        timeout=10
    )
