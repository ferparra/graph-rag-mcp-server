"""Expression evaluator for Obsidian .base computed fields."""

from __future__ import annotations
import re
import operator
from typing import Any, Dict, List, Callable, Union, cast
from datetime import datetime, date
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ExpressionContext(BaseModel):
    """Context for expression evaluation using Pydantic v2."""
    note_data: Dict[str, Any]
    computed_values: Dict[str, Any] = Field(default_factory=dict)


class ExpressionEvaluator:
    """Safe expression evaluator for computed fields.
    
    Implements a minimal, side-effect-free expression language supporting:
    - Basic operators: + - * / % == != > >= < <= && || !
    - Ternary operator: condition ? true_value : false_value
    - Functions: coalesce, clamp, lower, upper, trim, concat, contains, 
                 regexMatch, daysSince, now, date, len
    - Property access: note properties, file.* properties, computed references (@id)
    """
    
    def __init__(self):
        """Initialize the expression evaluator."""
        self.functions = self._init_functions()
        self.operators = self._init_operators()
    
    def _init_functions(self) -> Dict[str, Callable]:
        """Initialize built-in functions."""
        return {
            'coalesce': self._fn_coalesce,
            'clamp': self._fn_clamp,
            'lower': lambda s: str(s).lower() if s is not None else '',
            'upper': lambda s: str(s).upper() if s is not None else '',
            'trim': lambda s: str(s).strip() if s is not None else '',
            'concat': self._fn_concat,
            'contains': lambda haystack, needle: str(needle) in str(haystack),
            'regexMatch': self._fn_regex_match,
            'daysSince': self._fn_days_since,
            'now': lambda: datetime.now(),
            'date': self._fn_date,
            'len': lambda x: len(x) if hasattr(x, '__len__') else 0,
        }
    
    def _init_operators(self) -> Dict[str, Callable]:
        """Initialize operators."""
        return {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge,
            '<': operator.lt,
            '<=': operator.le,
            '&&': lambda a, b: bool(a) and bool(b),
            '||': lambda a, b: bool(a) or bool(b),
            '!': lambda a: not bool(a),
        }
    
    def evaluate(self, expression: str, context: ExpressionContext) -> Any:
        """Evaluate an expression in the given context.
        
        Args:
            expression: The expression string to evaluate
            context: The evaluation context with note data
            
        Returns:
            The evaluated result
            
        Raises:
            ValueError: If the expression is invalid or evaluation fails
        """
        try:
            # Tokenize the expression
            tokens = self._tokenize(expression)
            
            # Parse and evaluate
            result = self._evaluate_tokens(tokens, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {e}")
            raise ValueError(f"Expression evaluation failed: {e}")
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize an expression string.
        
        Args:
            expression: The expression to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenizer using regex
        # This handles identifiers, numbers, strings, operators, and punctuation
        pattern = r'''
            (?P<STRING>"[^"]*"|'[^']*')           |  # Strings
            (?P<NUMBER>\d+(?:\.\d+)?)              |  # Numbers
            (?P<IDENTIFIER>@?[a-zA-Z_][a-zA-Z0-9_-]*(?:\.[a-zA-Z_][a-zA-Z0-9_-]*)*)  |  # Identifiers (allow hyphens)
            (?P<OPERATOR>==|!=|<=|>=|&&|\|\||[+\-*/%<>!])      |  # Operators
            (?P<PUNCT>[(),?:])                     |  # Punctuation
            (?P<WHITESPACE>\s+)                       # Whitespace (ignored)
        '''
        
        tokens = []
        for match in re.finditer(pattern, expression, re.VERBOSE):
            kind = match.lastgroup
            value = match.group()
            
            if kind != 'WHITESPACE':
                tokens.append(value)
        
        return tokens
    
    def _evaluate_tokens(self, tokens: List[str], context: ExpressionContext) -> Any:
        """Evaluate tokenized expression.
        
        This is a simplified evaluator that handles basic expressions.
        For production, consider using a proper parser generator.
        
        Args:
            tokens: List of tokens
            context: Evaluation context
            
        Returns:
            Evaluated result
        """
        if not tokens:
            return None
        
        # Handle simple cases first
        if len(tokens) == 1:
            return self._evaluate_value(tokens[0], context)
        
        # Handle parenthesized expressions that wrap the whole token list
        if tokens[0] == '(':
            paren_depth = 0
            end_idx = -1
            for i, tok in enumerate(tokens):
                if tok == '(':
                    paren_depth += 1
                elif tok == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        end_idx = i
                        break
            # If the matching ')' is the last token, strip parentheses and evaluate inside
            if end_idx == len(tokens) - 1:
                return self._evaluate_tokens(tokens[1:-1], context)
        
        # Look for ternary operator (? :) at top level
        if '?' in tokens and ':' in tokens:
            paren_depth = 0
            q_idx = -1
            c_idx = -1
            for i, tok in enumerate(tokens):
                if tok == '(':
                    paren_depth += 1
                elif tok == ')':
                    paren_depth -= 1
                elif tok == '?' and paren_depth == 0:
                    q_idx = i
                    break
            if q_idx != -1:
                paren_depth = 0
                for j in range(q_idx + 1, len(tokens)):
                    tok = tokens[j]
                    if tok == '(':
                        paren_depth += 1
                    elif tok == ')':
                        paren_depth -= 1
                    elif tok == ':' and paren_depth == 0:
                        c_idx = j
                        break
            if q_idx != -1 and c_idx != -1:
                return self._evaluate_ternary(tokens, context)
        
        # Look for function calls first (handles arguments safely)
        if len(tokens) > 1 and tokens[1] == '(':
            # Ensure we only treat it as a function call if the call is at the beginning
            # Find the matching closing parenthesis for the opening at index 1
            paren_depth = 0
            end_idx = -1
            for i, tok in enumerate(tokens[1:], 1):
                if tok == '(':
                    paren_depth += 1
                elif tok == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                # Evaluate just the function call portion
                func_value = self._evaluate_function(tokens[:end_idx+1], context)
                # If there are remaining tokens (e.g., func(a) + b), evaluate the rest
                if end_idx + 1 < len(tokens):
                    # Build a new token list with the evaluated value as a literal
                    remainder = tokens[end_idx+1:]
                    value_token = str(func_value) if not isinstance(func_value, str) else f'"{func_value}"'
                    return self._evaluate_tokens([value_token] + remainder, context)
                return func_value

        # Handle unary operators
        if tokens[0] == '!':
            operand = self._evaluate_tokens(tokens[1:], context)
            return self.operators['!'](operand)
        if tokens[0] == '-' and len(tokens) > 1:
            # Unary minus
            operand = self._evaluate_tokens(tokens[1:], context)
            try:
                return -float(operand)
            except (TypeError, ValueError):
                return -0.0

        # Look for binary operators at top level (respect parentheses)
        for op_group in [['||'], ['&&'], ['==', '!=', '>', '>=', '<', '<='], ['+', '-'], ['*', '/', '%']]:
            paren_depth = 0
            for i, token in enumerate(tokens):
                if token == '(':
                    paren_depth += 1
                elif token == ')':
                    paren_depth -= 1
                elif paren_depth == 0 and token in op_group:
                    left = self._evaluate_tokens(tokens[:i], context)
                    right = self._evaluate_tokens(tokens[i+1:], context)
                    return self.operators[token](left, right)
        
        # Look for function calls (fallback)
        if len(tokens) > 1 and tokens[1] == '(':
            return self._evaluate_function(tokens, context)
        
        # Default: evaluate as single value
        return self._evaluate_value(' '.join(tokens), context)
    
    def _evaluate_value(self, token: str, context: ExpressionContext) -> Any:
        """Evaluate a single value token.
        
        Args:
            token: The token to evaluate
            context: Evaluation context
            
        Returns:
            The value
        """
        # String literal
        if (token.startswith('"') and token.endswith('"')) or \
           (token.startswith("'") and token.endswith("'")):
            return token[1:-1]
        
        # Number literal
        try:
            if '.' in token:
                return float(token)
            return int(token)
        except ValueError:
            pass
        
        # Boolean literals
        if token.lower() == 'true':
            return True
        if token.lower() == 'false':
            return False
        if token.lower() == 'null' or token.lower() == 'none':
            return None
        
        # Computed reference
        if token.startswith('@'):
            ref_id = token[1:]
            return context.computed_values.get(ref_id)
        
        # Property access (including dotted notation)
        if '.' in token:
            parts = token.split('.')
            node: Dict[str, Any] = context.note_data
            for part in parts[:-1]:
                next_val = node.get(part)
                if not isinstance(next_val, dict):
                    return None
                node = cast(Dict[str, Any], next_val)
            return node.get(parts[-1])
        
        # Simple property
        return context.note_data.get(token)
    
    def _evaluate_ternary(self, tokens: List[str], context: ExpressionContext) -> Any:
        """Evaluate ternary operator (condition ? true_val : false_val).
        
        Args:
            tokens: List of tokens containing ternary expression
            context: Evaluation context
            
        Returns:
            True or false value based on condition
        """
        # Find top-level '?' and matching ':' (ignore those inside parentheses)
        paren_depth = 0
        q_idx = -1
        c_idx = -1
        for i, tok in enumerate(tokens):
            if tok == '(':
                paren_depth += 1
            elif tok == ')':
                paren_depth -= 1
            elif tok == '?' and paren_depth == 0 and q_idx == -1:
                q_idx = i
                break
        if q_idx == -1:
            return self._evaluate_value(' '.join(tokens), context)
        paren_depth = 0
        for j in range(q_idx + 1, len(tokens)):
            tok = tokens[j]
            if tok == '(':
                paren_depth += 1
            elif tok == ')':
                paren_depth -= 1
            elif tok == ':' and paren_depth == 0:
                c_idx = j
                break
        if c_idx == -1:
            return self._evaluate_value(' '.join(tokens), context)

        condition = self._evaluate_tokens(tokens[:q_idx], context)
        true_val = self._evaluate_tokens(tokens[q_idx+1:c_idx], context)
        false_val = self._evaluate_tokens(tokens[c_idx+1:], context)
        
        return true_val if condition else false_val
    
    def _evaluate_function(self, tokens: List[str], context: ExpressionContext) -> Any:
        """Evaluate a function call.
        
        Args:
            tokens: List of tokens starting with function name
            context: Evaluation context
            
        Returns:
            Function result
        """
        func_name = tokens[0]
        
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        # Find matching parentheses
        paren_count = 0
        end_idx = 1
        for i, token in enumerate(tokens[1:], 1):
            if token == '(':
                paren_count += 1
            elif token == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = i
                    break
        
        # Extract and evaluate arguments
        arg_tokens = tokens[2:end_idx]  # Skip function name and opening paren
        args = self._parse_function_args(arg_tokens, context)
        
        # Call function
        func = self.functions[func_name]
        return func(*args)
    
    def _parse_function_args(self, tokens: List[str], context: ExpressionContext) -> List[Any]:
        """Parse function arguments from tokens.
        
        Args:
            tokens: Tokens between function parentheses
            context: Evaluation context
            
        Returns:
            List of evaluated arguments
        """
        if not tokens:
            return []
        
        args = []
        current_arg = []
        paren_depth = 0
        
        for token in tokens:
            if token == ',' and paren_depth == 0:
                # End of current argument
                if current_arg:
                    args.append(self._evaluate_tokens(current_arg, context))
                current_arg = []
            else:
                if token == '(':
                    paren_depth += 1
                elif token == ')':
                    paren_depth -= 1
                current_arg.append(token)
        
        # Add last argument
        if current_arg:
            args.append(self._evaluate_tokens(current_arg, context))
        
        return args
    
    # Built-in function implementations
    
    def _fn_coalesce(self, *args) -> Any:
        """Return first non-null argument."""
        for arg in args:
            if arg is not None:
                return arg
        return None
    
    def _fn_clamp(self, value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float]) -> Union[int, float]:
        """Clamp value between min and max."""
        try:
            value = float(value) if value is not None else 0
            min_val = float(min_val)
            max_val = float(max_val)
            return max(min_val, min(value, max_val))
        except (TypeError, ValueError):
            return 0
    
    def _fn_concat(self, *args) -> str:
        """Concatenate arguments as strings."""
        return ''.join(str(arg) if arg is not None else '' for arg in args)
    
    def _fn_regex_match(self, text: str, pattern: str) -> bool:
        """Check if text matches regex pattern."""
        try:
            return bool(re.search(pattern, str(text)))
        except re.error:
            return False
    
    def _fn_days_since(self, date_value: Any) -> int:
        """Calculate days since given date."""
        try:
            if isinstance(date_value, str):
                # Try parsing ISO date
                if 'T' in date_value:
                    dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(date_value, '%Y-%m-%d')
            elif isinstance(date_value, (date, datetime)):
                dt = date_value
            else:
                return 0
            
            if isinstance(dt, datetime):
                dt = dt.date()
            
            delta = date.today() - dt
            return delta.days
            
        except (ValueError, TypeError):
            return 0
    
    def _fn_date(self, format_str: str, source: Any) -> str:
        """Format a date value."""
        try:
            if isinstance(source, str):
                # Parse the date
                if 'T' in source:
                    dt = datetime.fromisoformat(source.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(source, '%Y-%m-%d')
            elif isinstance(source, (date, datetime)):
                dt = source
            else:
                return ''
            
            # Format according to format string
            return dt.strftime(format_str)
            
        except (ValueError, TypeError):
            return ''


class CircularReferenceError(Exception):
    """Raised when circular references are detected in computed fields."""
    pass


def evaluate_computed_fields(base_file: Any, note_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all computed fields for a note.
    
    Args:
        base_file: BaseFile object with computed field definitions
        note_data: Note data dictionary
        
    Returns:
        Dictionary of computed field values
        
    Raises:
        CircularReferenceError: If circular references are detected
    """
    evaluator = ExpressionEvaluator()
    context = ExpressionContext(note_data=note_data)
    computed = {}
    evaluated = set()
    evaluating = set()
    
    def evaluate_field(field_id: str) -> Any:
        """Recursively evaluate a computed field."""
        if field_id in evaluated:
            return computed[field_id]
        
        if field_id in evaluating:
            raise CircularReferenceError(f"Circular reference detected: {field_id}")
        
        # Find the field definition
        field_def = next((f for f in base_file.computed if f.id == field_id), None)
        if not field_def:
            return None
        
        evaluating.add(field_id)
        
        try:
            # Evaluate the expression
            result = evaluator.evaluate(field_def.expr, context)
            
            # Type coercion based on declared type
            if field_def.type == 'number':
                try:
                    result = float(result) if result is not None else 0
                except (TypeError, ValueError):
                    result = 0
            elif field_def.type == 'boolean':
                result = bool(result)
            elif field_def.type == 'string':
                result = str(result) if result is not None else ''
            
            computed[field_id] = result
            context.computed_values[field_id] = result
            evaluated.add(field_id)
            evaluating.remove(field_id)
            
            return result
            
        except Exception as e:
            evaluating.remove(field_id)
            raise ValueError(f"Failed to evaluate field '{field_id}': {e}")
    
    # Evaluate all computed fields
    for field in base_file.computed:
        if field.id not in evaluated:
            evaluate_field(field.id)
    
    return computed
