import json
import re

# Read the file
with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'r') as f:
    content = f.read()

# Find the position after _truncate_tool_result to add the new function
marker = '''def _truncate_tool_result(result: Any) -> str:
    """
    Hard-cap tool result string to 15000 characters.
    If truncated, append a note with the original length.
    """
    result_str = str(result)
    if len(result_str) <= 15000:
        return result_str
    original_len = len(result_str)
    return result_str[:15000] + f"\\n... (truncated from {original_len} chars)"'''

new_function = '''

def _parse_tool_args(arguments: str, fn_name: str) -> Dict[str, Any]:
    """
    Parse tool arguments from JSON string. If JSON parsing fails, attempt
    to extract arguments from common LLM output formats.
    
    Returns dict or TOOL_ARG_ERROR marker.
    """
    if not arguments or not arguments.strip():
        return {"error": "empty_arguments"}
    
    # Try standard JSON first
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try to extract JSON from within the string (handles markdown code blocks)
    json_match = re.search(r'\\{[^{}]*\\}', arguments)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Last resort: return empty dict to allow tool to run with defaults
    return {}


'''

if marker in content:
    content = content.replace(marker, marker + new_function)
    with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'w') as f:
        f.write(content)
    print('Done')
else:
    print('Marker not found')
