


import math
import json
import re
from typing import Dict, List, Optional
from enum import Enum

def extract_answer_logprobs(response, output_format: 'OutputFormat') -> Dict[str, str]:
    """
    Extract log probabilities for answer choices A, B, C, D from model response.
    Uses the OutputFormat enum to determine the appropriate parsing strategy.
    
    Args:
        response: The model response object containing logprobs data
        output_format: OutputFormat enum indicating the expected answer format
        
    Returns:
        Dictionary with probabilities as strings: {"A": "0.9992", "B": "0.0005", ...}
        Returns zeros for all choices if logprobs unavailable or parsing fails.
    """
    # Initialize default result with zeros as fallback
    default_result = {"A": "0", "B": "0", "C": "0", "D": "0"}
    
    try:
        # Extract logprobs data from response metadata
        logprobs_data = response.response_metadata.get('logprobs')
        if not logprobs_data or not logprobs_data.get('tokens'):
            print("No logprobs data found in response")
            return default_result
        
        tokens = logprobs_data['tokens']
        top_logprobs = logprobs_data.get('top_logprobs', [])
        
        # Select parsing strategy based on the OutputFormat enum
        answer_position = None
        if output_format.name.lower == "json_answer" or  output_format.name.lower == "json_full":
            # Handle JSON formats: {"Answer": "<B>"} or {"Reasoning": "...", "Answer": "<B>"}
            answer_position = _find_json_answer_position(tokens)
            
        elif output_format.name.lower == "xml_answer" or  output_format.name.lower == "xml_full":

            # Handle XML formats: <Answer>B</Answer> or <Answer>[B]</Answer>
            answer_position = _find_xml_answer_position(tokens)
            
        else:  # output_format.name.lower == "base"
            # Handle BASE format: Answer: B
            answer_position = _find_base_answer_position(tokens)
        
        # Implement graceful fallback: if format-specific parsing fails, try all methods
        # This handles cases where the model doesn't follow the expected format exactly
        if answer_position is None:
            print(f"Format-specific parsing failed for {output_format.name}, trying fallback methods...")
            answer_position = (_find_base_answer_position(tokens) or 
                             _find_json_answer_position(tokens) or 
                             _find_xml_answer_position(tokens))
        
        # Ultimate fallback: search for any A/B/C/D letter after the word "answer"
        # This handles creative model responses that don't follow any specific format
        if answer_position is None:
            print("All format-specific methods failed, using universal answer search...")
            answer_position = _find_answer_after_answer_keyword(tokens)
        
        # Validate that we found a valid position with available logprobs
        if answer_position is None or answer_position >= len(top_logprobs):
            print(f"Could not locate answer position in token sequence, Tokens: {tokens}")
            
            return default_result
        
        # Extract and convert the log probabilities at the identified position
        return _extract_probabilities_at_position(top_logprobs[answer_position])
        
    except Exception as e:
        print(f"Error extracting logprobs: {e}")
        return default_result


def _find_base_answer_position(tokens: List[str]) -> Optional[int]:
    """
    Find answer position in BASE format: 'Answer: B'
    
    Searches for the token sequence: ["Answer", ":", "B"] or ["Answer", ":", " B"]
    """
    for i in range(len(tokens) - 2):
        if (tokens[i] == 'Answer' and 
            tokens[i + 1] == ':' and 
            len(tokens) > i + 2):
            
            # Check immediate next token for answer letter
            next_token = tokens[i + 2].strip()
            if next_token in ['A', 'B', 'C', 'D']:
                return i + 2
                
            # Handle case where there's a space token between colon and letter
            elif len(tokens) > i + 3:
                next_next_token = tokens[i + 3].strip()
                if next_next_token in ['A', 'B', 'C', 'D']:
                    return i + 3
    return None


def _find_json_answer_position(tokens: List[str]) -> Optional[int]:
    """
    Find answer position in JSON formats: {"Answer": "<B>"} or {"Reasoning": "...", "Answer": "<B>"}
    
    Strategy: Locate the "Answer" key, then search for the letter value in surrounding tokens
    """
    # First, locate the "Answer" key in the JSON structure
    answer_key_pos = None
    for i, token in enumerate(tokens):
        # Handle different tokenization patterns for the JSON key
        if '"Answer"' in token or (token == 'Answer' and i > 0 and '"' in tokens[i-1]):
            answer_key_pos = i
            break
    
    if answer_key_pos is None:
        return None
    
    # Search for answer letters in a reasonable window after the "Answer" key
    # JSON structure means the value should appear within the next several tokens
    search_end = min(len(tokens), answer_key_pos + 10)
    
    for i in range(answer_key_pos + 1, search_end):
        token = tokens[i].strip()
        
        # Direct letter match (for cases where angle brackets are separate tokens)
        if token in ['A', 'B', 'C', 'D']:
            return i
        
        # Letter within angle brackets: "<A>", "<B>", etc.
        bracket_match = re.search(r'<([ABCD])>', token)
        if bracket_match:
            return i
        
        # Handle case where opening bracket is separate: "<" followed by letter
        if token.startswith('<') and len(tokens) > i + 1:
            next_token = tokens[i + 1].strip()
            if next_token in ['A', 'B', 'C', 'D']:
                return i + 1
    
    return None


def _find_xml_answer_position(tokens: List[str]) -> Optional[int]:
    """
    FIXED XML parser that handles the actual tokenization pattern.
    
    Real pattern we discovered:
    - Position N-1: " <" (space + opening bracket)
    - Position N: "Answer" 
    - Position N+1: ">D" or ">["
    - Position N+2: "D" (if bracketed)
    """
    print("🔍 Starting FIXED XML answer search...")
    
    for i in range(len(tokens)):
        if tokens[i] == 'Answer':
            print(f"📍 Found 'Answer' token at position {i}")
            
            # Get context tokens
            prev_token = tokens[i-1] if i > 0 else ""
            next_token = tokens[i+1] if i+1 < len(tokens) else ""
            
            print(f"📋 Context: prev='{prev_token}' | current='Answer' | next='{next_token}'")
            
            # Check for XML pattern: " <" + "Answer" + ">..."
            if (prev_token.strip().endswith('<') and 
                next_token.startswith('>')):
                
                print(f"✅ XML pattern detected!")
                
                # Extract content after the >
                content_after_bracket = next_token[1:]  # Remove the >
                print(f"📝 Content after '>': '{content_after_bracket}'")
                
                # Case 1: Direct answer like ">D"
                if content_after_bracket.strip() in ['A', 'B', 'C', 'D']:
                    print(f"🎯 Direct answer '{content_after_bracket.strip()}' found at position {i+1}")
                    return i + 1
                
                # Case 2: Bracketed answer like ">["
                elif content_after_bracket.strip() == '[' and i+2 < len(tokens):
                    letter_token = tokens[i+2]
                    print(f"🔍 Checking bracketed pattern, letter token: '{letter_token}'")
                    if letter_token.strip() in ['A', 'B', 'C', 'D']:
                        print(f"🎯 Bracketed answer '{letter_token.strip()}' found at position {i+2}")
                        return i + 2
                
                print(f"⚠️ XML pattern detected but no valid answer letter found")
            else:
                print(f"❌ Not XML pattern - prev doesn't end with '<' or next doesn't start with '>'")
    
    print("❌ No XML answer pattern found")
    return None


def _extract_probabilities_at_position(token_probs: Dict[str, float]) -> Dict[str, str]:
    """
    Convert log probabilities to regular probabilities for answer choices.
    
    Args:
        token_probs: Dictionary mapping token strings to their log probabilities
        
    Returns:
        Dictionary with answer choices as keys and formatted probability strings as values
    """
    result = {"A": "0", "B": "0", "C": "0", "D": "0"}
    
    for token, log_prob in token_probs.items():
        # Extract the answer letter from potentially complex token formats
        clean_letter = _extract_letter_from_token(token)
        
        if clean_letter in ['A', 'B', 'C', 'D']:
            # Convert from log space to probability space: exp(log_prob)
            regular_prob = math.exp(log_prob)
            # Format as string with 4 decimal places for consistency
            result[clean_letter] = f"{regular_prob:.4f}"
    
    return result


def _find_answer_after_answer_keyword(tokens: List[str]) -> Optional[int]:
    """
    Universal fallback method that searches for any A/B/C/D letter in various contexts.
    
    This comprehensive approach handles multiple scenarios that format-specific parsers might miss:
    - Natural language: "The answer is B" instead of "Answer: B"
    - XML in single token: "<Answer>B</Answer>" as one token
    - JSON in single token: '{"Answer": "<B>"}' as one token  
    - Mixed formats: "My answer would be C" with creative punctuation
    - Embedded patterns: Letters within larger structured tokens
    
    Strategy: 
    1. Look for "answer" keyword and search nearby tokens
    2. Check each token for embedded answer letters regardless of context
    3. Fall back to any isolated A/B/C/D letters
    """
    # Define comprehensive variations of "answer" including XML/JSON contexts
    # This covers different cases, punctuation, and structural variations
    answer_keywords = ['answer', 'Answer', 'ANSWER', 'answers', 'Answers']
    
    # First approach: Find "answer" keyword and search forward
    # This handles most natural language and semi-structured responses
    for i, token in enumerate(tokens):
        # Clean the token by removing common punctuation to improve matching
        cleaned_token = re.sub(r'[^\w]', '', token.lower())
        
        # Check if this token contains any variation of "answer"
        if any(keyword.lower() in cleaned_token for keyword in answer_keywords):
            print(f"Found answer keyword at position {i}: '{token}'")
            
            # First, check if the answer letter is embedded in the same token
            # This handles cases like "<Answer>B</Answer>" or '{"Answer":"<B>"}'
            current_token_letter = _extract_letter_from_token(token)
            if current_token_letter:
                print(f"Found embedded answer letter '{current_token_letter}' in keyword token '{token}' at position {i}")
                return i
            
            # Then search forward from this position for any answer letter
            # Use a reasonable search window to avoid false positives from distant letters
            search_window_end = min(len(tokens), i + 15)  # Look up to 15 tokens ahead
            
            for j in range(i + 1, search_window_end):
                candidate_token = tokens[j].strip()
                
                # Check for direct letter match
                if candidate_token in ['A', 'B', 'C', 'D']:
                    print(f"Found answer letter '{candidate_token}' at position {j}")
                    return j
                
                # Check for letters embedded in other tokens (like "choice_B" or "option-A")
                extracted_letter = _extract_letter_from_token(candidate_token)
                if extracted_letter:
                    print(f"Found embedded answer letter '{extracted_letter}' in token '{candidate_token}' at position {j}")
                    return j
    
    # Second approach: Comprehensive scan for structured patterns
    # This catches XML, JSON, and other structured formats even without "answer" keyword
    print("No answer keyword found, scanning all tokens for structured patterns...")
    for i, token in enumerate(tokens):
        # Look for comprehensive embedded patterns in each token
        extracted_letter = _extract_letter_from_token(token)
        if extracted_letter:
            print(f"Found structured answer pattern '{extracted_letter}' in token '{token}' at position {i}")
            return i
    
    # Third approach: Look for isolated A/B/C/D letters anywhere in the response
    # This is the most permissive fallback for minimal or creative responses
    print("No structured patterns found, searching for any isolated A/B/C/D letters...")
    for i, token in enumerate(tokens):
        if token.strip() in ['A', 'B', 'C', 'D']:
            print(f"Found isolated answer letter '{token}' at position {i}")
            return i
    
    print("Universal fallback: No answer letters found in any format")
    return None


def _extract_letter_from_token(token: str) -> str:
    """
    Extract answer letter from various token formats that might appear in different output formats.
    
    This function is designed to be maximally permissive and handle creative model outputs.
    It systematically checks multiple pattern types, from most specific to most general.
    
    Patterns handled:
    - Direct letters: "A", "B", "C", "D"
    - XML single-token: "<Answer>B</Answer>", "<answer>C</answer>"
    - JSON single-token: '{"Answer":"<B>"}', '{"answer":"D"}'
    - Brackets: "[A]", "(B)", "<C>", "{D}"
    - Option indicators: "option_A", "choice-B", "alternative:C"
    - Mixed formats: Any token containing A/B/C/D
    """
    if not token:
        return ""
    
    token = token.strip()
    
    # Most common case: direct letter match
    if token in ['A', 'B', 'C', 'D']:
        return token
    
    # Handle complete XML structures in single tokens: "<Answer>B</Answer>", "<answer>C</answer>"
    xml_complete_match = re.search(r'<[Aa]nswer>([ABCD])</[Aa]nswer>', token)
    if xml_complete_match:
        return xml_complete_match.group(1)
    
    # Handle XML with square brackets (XML_FULL format): "<Answer>[B]</Answer>"
    xml_bracket_match = re.search(r'<[Aa]nswer>\[([ABCD])\]</[Aa]nswer>', token)
    if xml_bracket_match:
        return xml_bracket_match.group(1)
    
    # Handle complete JSON structures in single tokens
    # Patterns: '{"Answer":"<B>"}', '{"answer":"D"}', '{"Answer":"B"}'
    json_complete_match = re.search(r'\{[^}]*[Aa]nswer[^}]*["\']<?([ABCD])>?["\'][^}]*\}', token)
    if json_complete_match:
        return json_complete_match.group(1)
    
    # Standard bracket patterns from our original implementation
    # JSON format: extract from angle brackets "<A>", "<B>", etc.
    bracket_match = re.search(r'<([ABCD])>', token)
    if bracket_match:
        return bracket_match.group(1)
    
    # XML_FULL format: extract from square brackets "[A]", "[B]", etc.
    square_match = re.search(r'\[([ABCD])\]', token)
    if square_match:
        return square_match.group(1)
    
    # Handle parentheses: "(A)", "(B)", etc. - common in natural language responses
    paren_match = re.search(r'\(([ABCD])\)', token)
    if paren_match:
        return paren_match.group(1)
    
    # Handle curly braces: "{A}", "{B}", etc. - sometimes used for emphasis
    curly_match = re.search(r'\{([ABCD])\}', token)
    if curly_match:
        return curly_match.group(1)
    
    # Handle option indicators: "option_A", "choice-B", "alternative:C", etc.
    option_match = re.search(r'(?:option|choice|alternative)[-_:\s]*([ABCD])', token, re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()
    
    # Handle common answer formats: "answer:A", "ans=B", "solution_C"
    answer_format_match = re.search(r'(?:answer|ans|solution)[-_:=\s]*([ABCD])', token, re.IGNORECASE)
    if answer_format_match:
        return answer_format_match.group(1).upper()
    
    # Most permissive fallback: if the token contains any answer letter, extract it
    # This handles creative formats we haven't anticipated
    # We check in reverse order (D, C, B, A) to handle cases where multiple letters appear
    for letter in ['D', 'C', 'B', 'A']:
        if letter in token:
            return letter
    
    return ""  # Return empty string if no letter found