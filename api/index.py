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
    """Generates numerical vectors for Upstash mapping using gemini-embedding-001."""
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent'
    
    # Optimization: Use targeted retrieval modes to increase semantic match precision
    task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    
    body = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "embedContentConfig": {
            "output_dimensionality": 768  # Locks footprint size to your Upstash index layout
        }
    }
    res = requests.post(url, json=body, headers={'X-goog-api-key': GEMINI_API_KEY}, timeout=15)
    return res.json()['embedding']['values']

# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Moose is alive 🐾'})

@app.route('/', methods=['POST'])
def slack_handler():
    """Handles conversational workspace user mentions."""
    payload = request.get_json(force=True, silent=True) or {}

    if payload.get('type') == 'url_verification':
        return jsonify({'challenge': payload.get('challenge')})

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

    # Guard execution safety limit caps
    if not check_gemini_rate_limit():
        post_slack_reply(channel, thread_ts, "⚠️ System rate limiter active. Ask again in an hour.")
        return jsonify({'ok': True})

    # Execute search against Upstash Index
    question_vector = get_embedding(question, is_query=True)
    search_results = vector_index.query(vector=question_vector, top_k=25, include_metadata=True)

    comments = []
    for res in search_results:
        comments.append({
            'date': res.metadata.get('date'),
            'playerCode': res.metadata.get('playerCode'),
            'comment': res.metadata.get('comment')
        })

    answer = answer_question(question, comments)
    post_slack_reply(channel, thread_ts, answer)
    return jsonify({'ok': True})


@app.route('/cache', methods=['POST'])
def cache_handler():
    """Receives newly tracked tickets to index patterns and handle alerts."""
    global last_alert_time
    
    auth = request.headers.get('X-Cache-Secret')
    if auth != CACHE_SECRET:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json(force=True, silent=True) or {}
    if not data or 'comment' not in data:
        return jsonify({'error': 'missing comment data'}), 400

    if not check_gemini_rate_limit():
        return jsonify({'error': 'Rate-limit buffer maxed out'}), 429

    comment_text = data['comment']
    player_code  = data.get('playerCode', 'unknown')
    comment_date = data.get('date', datetime.now().strftime('%b %d, %Y'))

    # Generate item vector
    comment_vector = get_embedding(comment_text, is_query=False)

    # Search for historical contextual matches
    matches = vector_index.query(vector=comment_vector, top_k=10, include_metadata=True)
    theme_matches = [m for m in matches if m.score >= 0.82] # 82%+ semantic pattern match
    
    # Save standard record entry directly inside Upstash Vector
    record_id = f"ticket_{datetime.now().timestamp()}"
    vector_index.upsert(
        vectors=[(record_id, comment_vector, {
            "comment": comment_text,
            "playerCode": player_code,
            "date": comment_date
        })]
    )

    # Evaluate dynamic threshold logic paired with spam cooldown protections
    now = time.time()
    if len(theme_matches) + 1 >= ALERT_THRESHOLD:
        if now - last_alert_time > ALERT_COOLDOWN_WINDOW:
            trigger_slack_alert(comment_text, len(theme_matches) + 1, player_code)
            last_alert_time = now
        else:
            print("ℹ️ Theme pattern threshold crossed, but system is inside alert spam cooldown window.")

    return jsonify({'ok': True, 'status': 'processed'})

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
    }
    res = requests.post(url, json=body, headers={'X-goog-api-key': GEMINI_API_KEY}, timeout=45)
    return res.json()['candidates'][0]['content']['parts'][0]['text']


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
