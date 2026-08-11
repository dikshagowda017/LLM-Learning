# ============================================================
# LLM Learning - Tokenization
# ============================================================
# Topic:
#   Tokenization in Large Language Models (LLMs)
#
# Concepts Covered:
#   1. Word Tokenization
#   2. Character Tokenization
#   3. Byte Tokenization
#
# Tokenization is the process of breaking text into smaller
# units called tokens, which can then be processed by an LLM.
# ============================================================

import re


def word_tokenization(text):
    # Split text into complete words and ignore punctuation
    return re.findall(r'\b\w+\b', text, re.UNICODE)


def character_tokenization(text):
    # Split text into individual characters
    return list(text)


def byte_tokenization(text):
    # Convert text into UTF-8 encoded bytes
    return list(text.encode('utf-8'))


# Example text
text = "Hello World!"

print("Word Tokens:", word_tokenization(text))
print("Character Tokens:", character_tokenization(text))
print("Byte Tokens:", byte_tokenization(text))