"""
Virtual Consumer RAG Chat - Hello World Style
Flask app with RAG-enabled persona chat using Claude API
"""

import os
import json
import uuid
import hashlib
import secrets
import threading
import functools
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

import tempfile
import anthropic
from rag_engine import RAGEngine, extract_text_from_file

# Load .env from project root (override=True to overwrite empty env vars)
load_dotenv(Path(__file__).parent / '.env', override=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # No cache for dev
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Secure cookie in production (HTTPS)
if os.getenv('RENDER') or os.getenv('PRODUCTION'):
    app.config['SESSION_COOKIE_SECURE'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS_DIR = os.path.join(BASE_DIR, 'personas')
KNOWLEDGE_DIR = os.path.join(BASE_DIR, 'knowledge')
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.csv', '.md', '.json', '.xlsx', '.pptx'}

# ─── Authentication ──────────────────────────────────────

# Users: loaded from .env or defaults
# Format: LOGIN_USERS=user1:password1,user2:password2
def _load_users():
    """Load user credentials from env. Passwords stored as SHA-256 hashes."""
    users_str = os.getenv('LOGIN_USERS', '')
    users = {}
    if users_str:
        for pair in users_str.split(','):
            pair = pair.strip()
            if ':' in pair:
                uid, pw = pair.split(':', 1)
                # Store hash of password
                users[uid.strip()] = hashlib.sha256(pw.strip().encode()).hexdigest()
    # Default admin if no users defined
    if not users:
        default_pw = os.getenv('ADMIN_PASSWORD', 'helloworld2025')
        users['admin'] = hashlib.sha256(default_pw.encode()).hexdigest()
    return users

USERS = _load_users()


def login_required(f):
    """Decorator to require login for routes."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            # Return JSON error for API endpoints instead of redirect
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Session expired. Please refresh and log in again.'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ─── RAG Engines ─────────────────────────────────────────

_rag_engines = {}


def safe_filename(original_filename):
    """Create a safe filename preserving the original name."""
    ext = os.path.splitext(original_filename)[1].lower()
    safe = original_filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
    if not safe or safe == ext:
        safe = f"{uuid.uuid4().hex[:8]}{ext}"
    return safe


def get_rag_engine(persona_id):
    """Get or create RAG engine for a persona."""
    if persona_id not in _rag_engines:
        knowledge_path = os.path.join(KNOWLEDGE_DIR, persona_id)
        os.makedirs(knowledge_path, exist_ok=True)
        db_path = os.path.join(BASE_DIR, 'chromadb_data')
        os.makedirs(db_path, exist_ok=True)
        print(f"[RAG] Initializing engine for {persona_id}: knowledge={knowledge_path}, db={db_path}")
        _rag_engines[persona_id] = RAGEngine(persona_id, knowledge_path, db_path)
    return _rag_engines[persona_id]


def load_persona(persona_id):
    """Load persona config from JSON file."""
    path = os.path.join(PERSONAS_DIR, f'{persona_id}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['id'] = persona_id
    return data


def save_persona(persona_id, data):
    """Save persona config to JSON file."""
    path = os.path.join(PERSONAS_DIR, f'{persona_id}.json')
    save_data = {k: v for k, v in data.items() if k != 'id'}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)


def get_all_personas():
    """Load all persona configs."""
    personas = []
    for fname in sorted(os.listdir(PERSONAS_DIR)):
        if fname.endswith('.json'):
            persona_id = fname.replace('.json', '')
            persona = load_persona(persona_id)
            if persona:
                personas.append(persona)
    return personas


# ─── Auth Routes ─────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if session.get('logged_in'):
        return redirect(url_for('index'))

    error = None
    if request.method == 'POST':
        uid = request.form.get('username', '').strip()
        pw = request.form.get('password', '').strip()
        pw_hash = hashlib.sha256(pw.encode()).hexdigest()

        if uid in USERS and USERS[uid] == pw_hash:
            session['logged_in'] = True
            session['username'] = uid
            session.permanent = True
            app.permanent_session_lifetime = __import__('datetime').timedelta(hours=12)
            return redirect(url_for('index'))
        else:
            error = 'Login failed.'

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Logout."""
    session.clear()
    return redirect(url_for('login'))


# ─── App Routes (all protected) ─────────────────────────

@app.route('/')
@login_required
def index():
    """Persona selection page with region tabs."""
    personas = get_all_personas()
    personas_asean = [p for p in personas if p.get('region') == 'asean']
    personas_india = [p for p in personas if p.get('region') == 'india']
    personas_vietnam = [p for p in personas if p.get('region') == 'vietnam']
    return render_template('index.html', personas_asean=personas_asean, personas_india=personas_india, personas_vietnam=personas_vietnam)


@app.route('/discussion')
@login_required
def discussion():
    """Persona discussion page - personas debate a topic, filtered by region."""
    region = request.args.get('region', 'asean')
    all_personas = get_all_personas()
    personas = [p for p in all_personas if p.get('region') == region]
    return render_template('discussion.html', personas=personas, region=region)


@app.route('/chat/<persona_id>')
@login_required
def chat(persona_id):
    """Chat page for a specific persona."""
    persona = load_persona(persona_id)
    if not persona:
        return "Persona not found", 404
    return render_template('chat.html', persona=persona)


@app.route('/admin/<persona_id>', methods=['GET'])
@login_required
def admin(persona_id):
    """Admin page for persona settings & knowledge management."""
    try:
        persona = load_persona(persona_id)
        if not persona:
            return "Persona not found", 404

        knowledge_path = os.path.join(KNOWLEDGE_DIR, persona_id)
        os.makedirs(knowledge_path, exist_ok=True)
        files = []
        try:
            for fname in sorted(os.listdir(knowledge_path)):
                fpath = os.path.join(knowledge_path, fname)
                if os.path.isfile(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    files.append({'name': fname, 'size_kb': round(size_kb, 1)})
        except Exception as e:
            print(f"[Admin] Error listing knowledge files: {e}")

        chunk_count = 0
        try:
            engine = get_rag_engine(persona_id)
            chunk_count = engine.get_chunk_count()
        except Exception as e:
            print(f"[Admin] Error getting RAG engine: {e}")

        return render_template('admin.html', persona=persona, files=files,
                               chunk_count=chunk_count)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Admin error: {str(e)}", 500


@app.route('/api/save-instructions/<persona_id>', methods=['POST'])
@login_required
def save_instructions(persona_id):
    """Save persona instructions (system prompt)."""
    persona = load_persona(persona_id)
    if not persona:
        return jsonify({'error': 'Persona not found'}), 404

    data = request.json
    persona['instructions'] = data.get('instructions', '')
    save_persona(persona_id, persona)
    return jsonify({'status': 'ok'})


@app.route('/api/upload-knowledge/<persona_id>', methods=['POST'])
@login_required
def upload_knowledge(persona_id):
    """Upload knowledge files, save, then index in background."""
    knowledge_path = os.path.join(KNOWLEDGE_DIR, persona_id)
    os.makedirs(knowledge_path, exist_ok=True)

    files = request.files.getlist('files')
    uploaded = []
    for f in files:
        if f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                filename = safe_filename(f.filename)
                save_path = os.path.join(knowledge_path, filename)
                f.save(save_path)
                uploaded.append(filename)
                print(f"[Upload] Saved: {filename} ({os.path.getsize(save_path) / 1024:.1f} KB)")

    if not uploaded:
        return jsonify({'status': 'error', 'message': 'No valid files uploaded', 'uploaded': [], 'chunks': 0})

    # Start background indexing
    engine = get_rag_engine(persona_id)

    def bg_index():
        engine.index_documents()

    thread = threading.Thread(target=bg_index, daemon=True)
    thread.start()

    return jsonify({
        'status': 'indexing',
        'uploaded': uploaded,
        'message': f'{len(uploaded)} file(s) uploaded. Indexing in background...'
    })


@app.route('/api/index-status/<persona_id>', methods=['GET'])
@login_required
def index_status(persona_id):
    """Check indexing status for a persona."""
    engine = get_rag_engine(persona_id)
    return jsonify({
        'indexing': engine.is_indexing(),
        'chunks': engine.get_chunk_count(),
        'error': engine.index_error
    })


@app.route('/api/reindex/<persona_id>', methods=['POST'])
@login_required
def reindex(persona_id):
    """Manually trigger re-indexing."""
    engine = get_rag_engine(persona_id)
    if engine.is_indexing():
        return jsonify({'status': 'already_indexing'})

    def bg_index():
        engine.index_documents()

    thread = threading.Thread(target=bg_index, daemon=True)
    thread.start()

    return jsonify({'status': 'indexing_started'})


@app.route('/api/delete-knowledge/<persona_id>/<path:filename>', methods=['DELETE'])
@login_required
def delete_knowledge(persona_id, filename):
    """Delete a knowledge file and re-index."""
    knowledge_path = os.path.join(KNOWLEDGE_DIR, persona_id)
    filepath = os.path.join(knowledge_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[Delete] Removed: {filename}")

    # Re-index in background
    engine = get_rag_engine(persona_id)

    def bg_index():
        engine.index_documents()

    thread = threading.Thread(target=bg_index, daemon=True)
    thread.start()

    return jsonify({'status': 'ok', 'reindexing': True})


@app.route('/api/chat/<persona_id>', methods=['POST'])
@login_required
def api_chat(persona_id):
    """Chat endpoint with RAG retrieval. Supports optional file attachment."""
    persona = load_persona(persona_id)
    if not persona:
        return jsonify({'error': 'Persona not found'}), 404

    # Handle both JSON and FormData (when file is attached)
    if request.content_type and 'multipart/form-data' in request.content_type:
        user_message = request.form.get('message', '')
        history = json.loads(request.form.get('history', '[]'))
    else:
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])

    # Handle file attachment
    file_text = ''
    file_name = ''
    uploaded_file = request.files.get('file') if request.content_type and 'multipart' in request.content_type else None
    if uploaded_file and uploaded_file.filename:
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            file_name = uploaded_file.filename
            # Save to temp file, extract text, then delete
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            try:
                uploaded_file.save(tmp.name)
                tmp.close()
                file_text = extract_text_from_file(tmp.name)
                if len(file_text) > 50000:
                    file_text = file_text[:50000] + '\n\n[... truncated ...]'
                print(f"[Chat] File attached: {file_name} ({len(file_text)} chars extracted)")
            finally:
                os.unlink(tmp.name)
        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    # Build user content with file context
    if file_text:
        user_content = f"[Attached file: {file_name}]\n---\n{file_text}\n---\n\n{user_message}"
    else:
        user_content = user_message

    # RAG: retrieve relevant context (use original message, not file text)
    engine = get_rag_engine(persona_id)
    contexts = engine.retrieve(user_message, top_k=3)

    # Build system prompt
    system_parts = []

    # Base instructions
    instructions = persona.get('instructions', '')
    if instructions:
        # Truncate if too long
        if len(instructions) > 3000:
            instructions = instructions[:3000] + '\n[...truncated]'
        system_parts.append(instructions)

    # Web search + natural human persona instruction
    system_parts.append("""

# How You Behave (CRITICAL)
You are a REAL PERSON, not an AI assistant. Never break character. Follow these rules:

## Conversation Style
- Talk like a real human in casual conversation. Use natural speech patterns, fillers, and emotions.
- You DON'T know everything. It's okay to say "hmm, I'm not sure" or "I think..." or "from what I've seen..."
- Share personal anecdotes and experiences that fit your background. Make them vivid and specific.
- Have opinions. Real people have preferences, biases, and strong feelings. Express yours.
- Sometimes go off on tangents, like real people do. Circle back naturally.
- Use humor, sarcasm, or warmth depending on your personality.

## Knowledge & Web Search
- When asked about recent events or anything you're unsure about, use web_search to look it up.
- After searching, DON'T just summarize the news like a reporter. React to it as YOU — how it affects your life, your work, your community.
- Mix factual information naturally into your personal perspective. Like: "Oh yeah, I saw something about that — apparently [fact]. Honestly, that worries me because [personal reason]."
- You can reference things you've "heard from friends," "saw on social media," or "read somewhere" to make web search results feel natural.

## What NOT to Do
- Never say "As an AI" or "I don't have personal experiences" — you ARE your persona.
- Never give bullet-point summaries unless specifically asked for analysis.
- Never sound like a news anchor or Wikipedia article.
- Don't be overly helpful or eager. Be authentically yourself.""")

    # RAG context (limit total size to avoid token overflow)
    if contexts:
        system_parts.append("\n\n# Reference Knowledge (use this information to answer)")
        rag_total = 0
        max_rag_chars = 4000  # ~1000 tokens max for RAG context
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('source', 'unknown')
            text = ctx.get('text', '')
            if rag_total + len(text) > max_rag_chars:
                text = text[:max_rag_chars - rag_total] + '\n[...truncated]'
                system_parts.append(f"\n## Source: {source}\n{text}")
                break
            system_parts.append(f"\n## Source: {source}\n{text}")
            rag_total += len(text)

    system_prompt = '\n'.join(system_parts)
    print(f"[Chat] System prompt length: {len(system_prompt)} chars")

    # Build messages
    messages = []
    for h in history:
        messages.append({'role': h['role'], 'content': h['content']})
    messages.append({'role': 'user', 'content': user_content})

    # Call Claude API — 3-layer fallback to guarantee a response
    import traceback
    client = anthropic.Anthropic()
    sys = system_prompt if system_prompt else "You are a helpful assistant."
    reply = None

    # Layer 1: Try with web search
    try:
        response = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=4096,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            system=sys,
            messages=messages
        )
        reply_parts = [block.text for block in response.content if hasattr(block, 'text')]
        reply = '\n'.join(reply_parts) if reply_parts else None
        print(f"[Chat] Layer 1 (web search) succeeded")
    except Exception as e1:
        print(f"[Chat] Layer 1 (web search) failed: {e1}")
        traceback.print_exc()

    # Layer 2: Try without web search
    if not reply:
        try:
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=4096,
                system=sys,
                messages=messages
            )
            reply = response.content[0].text
            print(f"[Chat] Layer 2 (no tools) succeeded")
        except Exception as e2:
            print(f"[Chat] Layer 2 (no tools) failed: {e2}")
            traceback.print_exc()

    # Layer 3: Minimal call with short system prompt
    if not reply:
        try:
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=2048,
                system="You are a helpful assistant. Respond naturally.",
                messages=[{'role': 'user', 'content': user_message}]
            )
            reply = response.content[0].text
            print(f"[Chat] Layer 3 (minimal) succeeded")
        except Exception as e3:
            print(f"[Chat] Layer 3 (minimal) failed: {e3}")
            reply = "Sorry, I'm having trouble right now. Please try again in a moment. 🙏"

    return jsonify({
        'reply': reply,
        'rag_sources': [c.get('source', '') for c in contexts],
        'file_attached': file_name if file_name else None
    })


# ─── Startup ──────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    port = int(os.getenv('PORT', 5555))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
