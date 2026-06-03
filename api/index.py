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
    is_reply  = event.get('thread_ts') and event.get('thread_ts') != event.get('ts')

    # Load comments from KV cache (most recent 50 only for speed)
    comments = kv_get()[:50]

    # Fetch thread history for context if this is a follow-up
    thread_history = get_thread_history(channel, thread_ts) if is_reply else []

    # Ask Gemini
    answer = answer_question(question, comments, thread_history)

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

def get_thread_history(channel, thread_ts):
    """Fetch previous messages in a Slack thread for conversation context."""
    res = requests.get(
        'https://slack.com/api/conversations.replies',
        headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
        params={'channel': channel, 'ts': thread_ts, 'limit': 20},
        timeout=10,
    )
    data = res.json()
    messages = data.get('messages', [])
    history = []
    for msg in messages:
        # Skip the bot_id check differently — include both user and bot messages
        text = re.sub(r'<@[A-Z0-9]+>\s*', '', msg.get('text', '')).strip()
        if not text:
            continue
        role = 'model' if msg.get('bot_id') else 'user'
        history.append({'role': role, 'parts': [{'text': text}]})
    return history


def answer_question(question, comments, thread_history=None):
    if comments:
        data_block = '\n'.join(
            f"{i+1}. [{c.get('date','?')}] [Support Code: {c.get('playerCode') or c.get('player_code','?')}] {c.get('comment','')}"
            for i, c in enumerate(comments)
        )
    else:
        data_block = f'(No negative feedback found in the last {LOOKBACK_DAYS} days)'

    prompt = f"""You are a CS intelligence assistant for a mobile game team. Answer questions about negative player feedback directly and thoroughly.

Feedback data ({len(comments)} negative comments, last {LOOKBACK_DAYS} days):
Each entry format: [date] [Support Code: unique player identifier used by CS team] comment
{data_block}

Today: {datetime.now().strftime('%b %-d, %Y')}

Question: "{question}"

Format rules:
- Start your answer immediately — no greetings, no "Moose here", no preamble
- Use * for bold, _ for italic. Never use ** double asterisks
- Bullet points must start with • not * or -
- No numbered lists, no hashtags, no parentheses as labels
- Always include: what the issue is, how many players, date range, and what players are saying
- Only report what the data actually shows — do not infer, speculate, or extrapolate beyond what players explicitly wrote
- If only 1 player reported something, say so clearly and do not imply it's a wider trend
- Keep the total response under 800 characters
- If there's no relevant data, say so in one sentence"""

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'

    # Build conversation: system prompt first, then thread history, then current question
    contents = [{'role': 'user', 'parts': [{'text': prompt}]},
                {'role': 'model', 'parts': [{'text': 'Understood. I will answer questions about this feedback data accurately and concisely.'}]}]

    if thread_history:
        # Append prior thread turns (skip the first system exchange)
        contents.extend(thread_history[1:])  # skip the original question already in prompt

    # Add current question as final user turn
    contents.append({'role': 'user', 'parts': [{'text': question}]})

    body = {
        'contents': contents,
        'generationConfig': {'maxOutputTokens': 2048},
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
    # Split into chunks of 3800 chars to stay under Slack's 4000 char limit
    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)]
    for chunk in chunks:
        requests.post(
            'https://slack.com/api/chat.postMessage',
            headers={'Authorization': f'Bearer {SLACK_BOT_TOKEN}'},
            json={
                'channel':   channel,
                'thread_ts': thread_ts,
                'text':      chunk,
                'username':  MOOSE_NAME,
                'icon_url':  MOOSE_ICON_URL,
            },
            timeout=10,
        )
