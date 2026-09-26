from tokenizers import Tokenizer

# Load a pretrained tokenizer (downloads once, caches locally)
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you today?"

# Encode: text -> token ids
encoded = tokenizer.encode(text)
print("Tokens:", encoded.tokens)
print("IDs:   ", encoded.ids)

# Decode: token ids -> text
decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)

