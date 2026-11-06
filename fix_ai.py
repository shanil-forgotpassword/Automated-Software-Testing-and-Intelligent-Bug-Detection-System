with open(r"c:\Users\vijay\Desktop\code guard ai\codeguard\ai.py", "r", encoding="utf-8") as f:
    content = f.read()

# Use tokenize to find the exact location of the syntax error
import io
import tokenize

try:
    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    print("Tokenization succeeded - file might be valid")
except tokenize.TokenError as e:
    print(f"Token error at line {e.args[1][0] if len(e.args) > 1 else 'unknown'}: {e}")
    
# Iterative fix: try to parse, find error, fix, repeat
import ast
max_attempts = 50
for attempt in range(max_attempts):
    try:
        ast.parse(content)
        print(f"Syntax OK after {attempt} fix attempts")
        with open(r"c:\Users\vijay\Desktop\code guard ai\codeguard\ai.py", "w", encoding="utf-8") as f:
            f.write(content)
        print("File written successfully")
        break
    except SyntaxError as e:
        if e.lineno is None:
            print(f"Syntax error location is unknown: {e}")
            break
        if attempt == max_attempts - 1:
            print(f"FAILED after {max_attempts} attempts. Error: line {e.lineno}: {e.msg}")
            lines = content.split('\n')
            for j in range(max(0, e.lineno-5), min(len(lines), e.lineno+5)):
                print(f"  {j+1}: {repr(lines[j])}")
            break
        
        lines = content.split('\n')
        err_line = e.lineno - 1
        err_text = lines[err_line]
        
        # Count unescaped quotes
        # A quote is unescaped if it's not preceded by an odd number of backslashes
        def count_unescaped_quotes(s):
            count = 0
            i = 0
            while i < len(s):
                if s[i] == '\\':
                    i += 2  # Skip escaped character
                    continue
                if s[i] == '"':
                    count += 1
                i += 1
            return count
        
        q_count = count_unescaped_quotes(err_text)
        
        if q_count % 2 == 1:
            found = False
            for j in range(err_line + 1, min(err_line + 10, len(lines))):
                next_text = lines[j]
                next_q = count_unescaped_quotes(next_text)
                next_stripped = next_text.strip()
                
                if next_stripped == '':
                    continue
                elif next_q % 2 == 1:
                    # Next line also has odd quotes - they pair up
                    lines[err_line] = err_text.rstrip() + '\\n' + next_text.lstrip()
                    lines[j] = ''
                    found = True
                    break
                elif next_q % 2 == 0 and next_q > 0:
                    # Next line has even quotes - could be closing
                    lines[err_line] = err_text.rstrip() + '\\n' + next_text.lstrip()
                    lines[j] = ''
                    found = True
                    break
                else:
                    break
            
            if not found:
                print(f"  Could not fix line {err_line + 1}: {repr(err_text)}")
                break
        else:
            print(f"  Line {err_line + 1} has even unescaped quotes: {e.msg}")
            # Show context
            for j in range(max(0, err_line-2), min(len(lines), err_line+3)):
                print(f"  {j+1}: {repr(lines[j])}")
            break
        
        content = '\n'.join(lines)
