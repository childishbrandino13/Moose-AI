from flask import Flask, request, jsonify
import requests
import os
import re
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# ── Config from environment variables (set in Vercel dashboard) ──
SLACK_BOT_TOKEN  = os.environ.get('SLACK_BOT_TOKEN')
GEMINI_API_KEY   = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL     = os.environ.get('GEMINI_MODEL', 'gemini-flash-latest')
MOOSE_NAME       = os.environ.get('MOOSE_NAME', 'Moose')
MOOSE_ICON_URL   = os.environ.get('MOOSE_ICON_URL', '')
LOOKBACK_DAYS    = int(os.environ.get('LOOKBACK_DAYS', '90'))
CACHE_SECRET     = os.environ.get('CACHE_SECRET')        # shared secret between Apps Script and Vercel

# Vercel KV (auto-populated when you add a KV store in Vercel dashboard)
KV_REST_API_URL   = os.environ.get('KV_REST_API_URL')
KV_REST_API_TOKEN = os.environ.get('KV_REST_API_TOKEN')
KV_KEY            = 'moose_comments'


# ── Routes ────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Moose is alive 🐾'})


@app.route('/', methods=['POST'])
def slack_handler():
    payload = request.get_json(force=True, silent=True) or {}

    # Slack URL verification
    if payload.get('type') == 'url_verification':
        return jsonify({'challenge': payload.get('challenge')})

    # Ignore retries (Slack retries if we're slow — avoid duplicate replies)
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

    # Load comments from KV cache (most recent 50 only for speed)
    comments = kv_get()[:50]

    # Ask Gemini
    answer = answer_question(question, comments)

    # Reply in Slack thread
    post_slack_reply(channel, thread_ts, answer)

    return jsonify({'ok': True})


@app.route('/cache', methods=['POST'])
def cache_handler():
    """
    Receives comment data pushed from Apps Script.
    Protected by a shared secret token.
    """
    auth = request.headers.get('X-Cache-Secret')
    if auth != CACHE_SECRET:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json(force=True, silent=True)
    if not data or 'comments' not in data:
        return jsonify({'error': 'missing comments'}), 400

    kv_set(data['comments'])
    return jsonify({'ok': True, 'count': len(data['comments'])})


# ── Vercel KV ─────────────────────────────────────────────────

def kv_set(comments):
    """Save comments array to KV store."""
    value = json.dumps(comments)
    requests.post(
        f'{KV_REST_API_URL}/set/{KV_KEY}',
        headers={
            'Authorization': f'Bearer {KV_REST_API_TOKEN}',
            'Content-Type':  'application/json',
        },
        data=value,
        timeout=10,
    )


def kv_get():
    """Read comments array from KV store."""
    res = requests.get(
        f'{KV_REST_API_URL}/get/{KV_KEY}',
        headers={'Authorization': f'Bearer {KV_REST_API_TOKEN}'},
        timeout=10,
    )
    data = res.json()
    result = data.get('result')
    if not result:
        return []
    try:
        return json.loads(result)
    except Exception:
        return []


# ── Gemini ────────────────────────────────────────────────────

def answer_question(question, comments):
    if comments:
        data_block = '\n'.join(
            f"{i+1}. [{c.get('date','?')}] [Player: {c.get('player_code','?')}] {c.get('comment','')}"
            for i, c in enumerate(comments)
        )
    else:
        data_block = f'(No negative feedback found in the last {LOOKBACK_DAYS} days)'

    prompt = f"""You are Moose 🐾, a friendly but data-driven CS intelligence assistant for a mobile game team. You have access to the last {LOOKBACK_DAYS} days of negative player feedback.

Feedback data ({len(comments)} negative comments):
{data_block}

Today's date: {datetime.now().strftime('%b %-d, %Y')}

Team question: "{question}"

Answer helpfully and concisely. Always include:
• Direct answer to the question (lead with this)
• Number of players affected (if relevant)
• Date range of the feedback (oldest → most recent) and how stale it is
• Brief summary of what players are saying
• Whether the issue appears to be growing, stable, or fading based on the dates
• If no relevant data exists, say so clearly

Use Slack markdown: *bold* for key numbers/labels, _italic_ for dates. Keep it scannable."""

    url  = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
    body = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': 1024},
    }
    res  = requests.post(
        url, json=body,
        headers={'X-goog-api-key': GEMINI_API_KEY},
        timeout=55,
    )
    data = res.json()

    if 'error' in data:
        return f"Sorry, I hit an error: {data['error'].get('message', 'unknown')}"

    return data['candidates'][0]['content']['parts'][0]['text']


# ── Slack ─────────────────────────────────────────────────────

def post_slack_reply(channel, thread_ts, text):
    requests.post(
        'https://slack.com/api/chat.postMessage',
        headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
        json={
            'channel':   channel,
            'thread_ts': thread_ts,
            'text':      text,
            'username':  MOOSE_NAME,
            'icon_url':  MOOSE_ICON_URL,
        },
        timeout=10,
    )
