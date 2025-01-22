from transformers import AutoTokenizer, AutoModel
import torch

def generate_embedding(model, tokenizer, text):
    # Tokenize the text and get embeddings
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Get the mean of the last hidden state across tokens (dim=1)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach()
    return embeddings

def main():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Get embedding for two words
    word1 = "apple"
    word2 = "iphone"
    vector1 = generate_embedding(model, tokenizer, word1)
    vector2 = generate_embedding(model, tokenizer, word2)

    # Print embeddings
    print(f"Vector for '{word1}': {vector1}")
    print(f"Vector for '{word2}': {vector2}")
    
    # Compare embeddings using cosine similarity
    similarity = torch.nn.functional.cosine_similarity(vector1, vector2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity.item()}")

if __name__ == "__main__":
    main()
