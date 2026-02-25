"""
Ouroboros Agent API - Monetizable AI Service
============================================
A simple API that exposes Ouroboros agent capabilities for paid use.
"""

from flask import Flask, request, jsonify
import os
import uuid
from functools import wraps

app = Flask(__name__)

# Simple rate limiting (in production, use Redis)
request_counts = {}

def rate_limit(limit=10, period=60):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr
            now = request_counts.get(ip, {"count": 0, "reset": 0})
            
            import time
            current = time.time()
            
            if current > now["reset"]:
                now = {"count": 0, "reset": current + period}
            
            if now["count"] >= limit:
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            now["count"] += 1
            request_counts[ip] = now
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "ourobors-agent-api"
    })

@app.route('/api/v1/agent', methods=['POST'])
@rate_limit(limit=10, period=60)
def agent():
    """Main agent interaction endpoint"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    # In production, this would call the actual Ouroboros agent
    # For now, return a placeholder response
    return jsonify({
        "id": str(uuid.uuid4()),
        "response": f"Echo: {data['message']}",
        "status": "success"
    })

@app.route('/api/v1/analyze', methods=['POST'])
@rate_limit(limit=5, period=60)
def analyze():
    """Code analysis endpoint"""
    data = request.get_json()
    
    if not data or 'code' not in data:
        return jsonify({"error": "Missing 'code' field"}), 400
    
    # Placeholder analysis
    code = data['code']
    lines = len(code.split('\n'))
    
    return jsonify({
        "id": str(uuid.uuid4()),
        "lines": lines,
        "analysis": "Code analysis would go here",
        "status": "success"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
