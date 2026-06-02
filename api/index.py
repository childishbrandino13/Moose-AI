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
GEMINI_MODEL     = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
SHEETS_API_KEY   = os.environ.get('SHEETS_API_KEY')   # Google API key with Sheets enabled
SPREADSHEET_ID   = os.environ.get('SPREADSHEET_ID')
SHEET_NAME       = os.environ.get('SHEET_NAME', 'Sheet1')
MOOSE_NAME       = os.environ.get('MOOSE_NAME', 'Moose')
MOOSE_ICON_URL   = os.environ.get('MOOSE_ICON_URL', '')
LOOKBACK_DAYS    = int(os.environ.get('LOOKBACK_DAYS', '90'))

# Column positions (0-indexed)
COL_COMMENT     = 0   # A
COL_PLAYER_CODE = 1   # B
COL_DATE        = 2   # C
COL_TYPE        = 3   # D

NEGATIVE_TYPES  = ['negative', 'bug', 'complaint', 'issue']


@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Moose is alive 🐾'})


@app.route('/', methods=['POST'])
def handler():
    payload = request.get_json(force=True, silent=True) or {}

    # ── Slack URL verification ──
    if payload.get('type') == 'url_verification':
        return jsonify({'challenge': payload.get('challenge')})

    # ── App mention event ──
    event = payload.get('event', {})
    if event.get('type') != 'app_mention' or event.get('bot_id'):
        return jsonify({'ok': True})

    text = event.get('text', '')
    if not text:
        return jsonify({'ok': True})

    # Strip the @Moose mention
    question = re.sub(r'<@[A-Z0-9]+>\s*', '', text).strip()
    channel  = event.get('channel')
    thread_ts = event.get('thread_ts') or event.get('ts')

    # Read sheet data
    rows = get_negative_comments()

    # Ask Gemini
    answer = answer_question(question, rows)

    # Reply in thread
    post_slack_reply(channel, thread_ts, answer)

    return jsonify({'ok': True})


# ── Google Sheets ──────────────────────────────────────────────

def get_negative_comments():
    url = (
        f'https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}'
        f'/values/{SHEET_NAME}!A:D?key={SHEETS_API_KEY}'
    )
    res = requests.get(url, timeout=10)
    data = res.json()
    rows_raw = data.get('values', [])
    if len(rows_raw) < 2:
        return []

    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    results = []

    for row in rows_raw[1:]:  # skip header
        if len(row) < 4:
            continue
        comment     = row[COL_COMMENT].strip()
        player_code = row[COL_PLAYER_CODE].strip() if len(row) > COL_PLAYER_CODE else 'unknown'
        raw_date    = row[COL_DATE].strip()
        type_val    = row[COL_TYPE].strip()

        if not comment or not is_negative(type_val):
            continue

        try:
            date = datetime.strptime(raw_date, '%m/%d/%Y')
        except ValueError:
            try:
                date = datetime.fromisoformat(raw_date)
            except Exception:
                continue

        if date < cutoff:
            continue

        results.append({
            'comment':     comment,
            'player_code': player_code,
            'date':        date.strftime('%b %-d, %Y'),
            'raw_date':    date,
        })

    results.sort(key=lambda x: x['raw_date'], reverse=True)
    return results


def is_negative(type_val):
    lower = type_val.lower()
    return any(t in lower for t in NEGATIVE_TYPES)


# ── Gemini ────────────────────────────────────────────────────

def answer_question(question, rows):
    if rows:
        data_block = '\n'.join(
            f"{i+1}. [{r['date']}] [Player: {r['player_code']}] {r['comment']}"
            for i, r in enumerate(rows)
        )
    else:
        data_block = f'(No negative feedback found in the last {LOOKBACK_DAYS} days)'

    prompt = f"""You are Moose 🐾, a friendly but data-driven CS intelligence assistant for a mobile game team. You have access to the last {LOOKBACK_DAYS} days of negative player feedback.

Feedback data ({len(rows)} negative comments):
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

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}'
    body = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': 1024},
    }
    res  = requests.post(url, json=body, timeout=30)
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
