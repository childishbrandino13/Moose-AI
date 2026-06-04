from flask import Flask, request, jsonify
import requests
import os
import re
from datetime import datetime
from upstash_vector import Index # Run: pip install upstash-vector

app = Flask(__name__)

# ── Config from environment variables ──
SLACK_BOT_TOKEN    = os.environ.get('SLACK_BOT_TOKEN')
GEMINI_API_KEY     = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL       = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
SLACK_ALERT_CHANNEL= os.environ.get('SLACK_ALERT_CHANNEL', 'C0B8581VDU')
CACHE_SECRET       = os.environ.get('CACHE_SECRET')
ALERT_THRESHOLD    = int(os.environ.get('ALERT_THRESHOLD', '3'))

# Upstash Vector Configuration
VECTOR_URL         = os.environ.get('UPSTASH_VECTOR_REST_URL')
VECTOR_TOKEN       = os.environ.get('UPSTASH_VECTOR_REST_TOKEN')
vector_index       = Index(url=VECTOR_URL, token=VECTOR_TOKEN)

# ── Helper: Generate Gemini Embeddings ────────────────────────
def get_embedding(text):
    # Using the universally supported gemini-embedding-001 endpoint
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent'
    body = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        # Safety fit: Locks the new model into your 768 Upstash dimension size
        "embedContentConfig": {
            "output_dimensionality": 768
        }
    }
    res = requests.post(url, json=body, headers={'X-goog-api-key': GEMINI_API_KEY}, timeout=15)
    return res.json()['embedding']['values']

# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['POST'])
def slack_handler():
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

    # Get Question Embedding Vector
    question_vector = get_embedding(question)

    # Search Vector DB for top 25 contextually matching complaints
    search_results = vector_index.query(
        vector=question_vector,
        top_k=25,
        include_metadata=True
    )

    # Map database matches into a clean string context block for Gemini
    comments = []
    for res in search_results:
        comments.append({
            'date': res.metadata.get('date'),
            'playerCode': res.metadata.get('playerCode'),
            'comment': res.metadata.get('comment')
        })

    # Ask Gemini based on targeted vector results
    answer = answer_question(question, comments)

    # Reply directly in the Slack thread
    post_slack_reply(channel, thread_ts, answer)
    return jsonify({'ok': True})


@app.route('/cache', methods=['POST'])
def cache_handler():
    """
    Receives single new comment row data pushed from Apps Script.
    Analyzes theme clustering patterns via vector search.
    """
    auth = request.headers.get('X-Cache-Secret')
    if auth != CACHE_SECRET:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json(force=True, silent=True)
    if not data or 'comment' not in data:
        return jsonify({'error': 'missing comment data'}), 400

    comment_text = data['comment']
    player_code  = data.get('playerCode', 'unknown')
    comment_date = data.get('date', datetime.now().strftime('%b %d, %Y'))

    # Generate vector embedding for incoming single comment
    comment_vector = get_embedding(comment_text)

    # Query for structurally identical historical records (Threshold score 0.82)
    matches = vector_index.query(
        vector=comment_vector,
        top_k=10,
        include_metadata=True
    )

    theme_matches = [m for m in matches if m.score >= 0.82]
    
    # Save new record to Upstash Vector Index
    record_id = f"ticket_{datetime.now().timestamp()}"
    vector_index.upsert(
        vectors=[
            (record_id, comment_vector, {
                "comment": comment_text,
                "playerCode": player_code,
                "date": comment_date
            })
        ]
    )

    # If matching theme patterns cross the tracking threshold, flag it in Slack
    if len(theme_matches) + 1 >= ALERT_THRESHOLD:
        trigger_slack_alert(comment_text, len(theme_matches) + 1, player_code)

    return jsonify({'ok': True, 'status': 'processed'})


def answer_question(question, comments):
    if comments:
        data_block = '\n'.join(
            f"{i+1}. [{c['date']}] [Support Code: {c['playerCode']}] {c['comment']}"
            for i, c in enumerate(comments)
        )
    else:
        data_block = '(No relevant historical customer complaints found in database matches)'

    prompt = f"""You are a CS intelligence assistant for a mobile sweeps app game team. Answer questions about negative player feedback directly and thoroughly.

Relevant historical context retrieved from database vector search:
{data_block}

Today: {datetime.now().strftime('%b %d, %Y')}
Question: "{question}"

Format rules:
- Start your answer immediately — no greetings or preambles
- Use * for bold, _ for italic.
- Bullet points must start with •
- Keep total response under 800 characters."""

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
