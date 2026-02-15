"""
Image Caption Generator - Streamlit Application
Loads a trained ResNet50 + LSTM model to generate captions for uploaded images
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
from collections import Counter

# ============================================================================
# CRITICAL: Copy your class definitions from the notebook
# ============================================================================

class Vocabulary:
    """Vocabulary class - MUST match your notebook implementation exactly"""
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.word_freq = Counter()
    
    def build_vocabulary(self, captions):
        """Build vocabulary from caption list."""
        for caption in captions:
            tokens = caption.split()
            self.word_freq.update(tokens)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, caption):
        """Convert caption to sequence of indices."""
        tokens = caption.split()
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
    
    def decode(self, indices):
        """Convert sequence of indices to caption."""
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)


class ImageEncoder(nn.Module):
    """
    ResNet50-based encoder for extracting image features
    CRITICAL: This MUST match your training architecture exactly
    """
    def __init__(self, encoded_image_size=7):
        super(ImageEncoder, self).__init__()
        # Load ResNet50 (no pretrained weights needed for inference)
        resnet = models.resnet50(pretrained=False)
        # Remove last two layers (avgpool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.encoded_image_size = encoded_image_size
        
    def forward(self, images):
        """
        Forward pass
        Input: (batch, 3, 224, 224)
        Output: (batch, encoded_size, encoded_size, 2048)
        """
        features = self.resnet(images)  # (batch, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (batch, 7, 7, 2048)
        return features


class Attention(nn.Module):
    """Attention mechanism for focusing on relevant image regions"""
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, hidden_state):
        """
        encoder_out: (batch, num_pixels, encoder_dim)
        hidden_state: (batch, hidden_dim)
        Returns: attention_weighted_encoding, attention_weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(hidden_state)  # (batch, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    LSTM decoder with attention mechanism
    CRITICAL: This MUST match your training architecture exactly
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encoder_dim=2048, 
                 attention_dim=512, num_layers=2, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # LSTM takes embedding + encoder_dim as input
        self.lstm = nn.LSTM(
            embedding_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        
    def init_hidden_state(self, encoder_out):
        """Initialize LSTM hidden state from encoder output"""
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch, hidden_dim)
        c = self.init_c(mean_encoder_out)  # (batch, hidden_dim)
        
        # Repeat for num_layers
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return h, c
    
    def forward(self, encoder_out, captions, lengths):
        """
        Training forward pass
        encoder_out: (batch, num_pixels, encoder_dim)
        captions: (batch, max_caption_length)
        lengths: actual caption lengths
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        
        # Flatten encoder output
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Initialize hidden state
        h, c = self.init_hidden_state(encoder_out)
        
        # Embed captions
        embeddings = self.embedding(captions)  # (batch, max_length, embedding_dim)
        
        # Initialize predictions tensor
        predictions = torch.zeros(batch_size, max(lengths), vocab_size).to(encoder_out.device)
        
        # Generate caption word by word
        for t in range(max(lengths)):
            # Get current hidden state (last layer)
            hidden_state = h[-1]  # (batch, hidden_dim)
            
            # Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out, hidden_state)
            
            # Concatenate embedding and attention
            lstm_input = torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1)
            lstm_input = lstm_input.unsqueeze(1)  # (batch, 1, embedding_dim + encoder_dim)
            
            # LSTM step
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Predict next word
            preds = self.fc(self.dropout(lstm_out.squeeze(1)))  # (batch, vocab_size)
            predictions[:, t, :] = preds
        
        return predictions


# ============================================================================
# Model Loading Functions
# ============================================================================

@st.cache_resource
def load_vocabulary():
    """Load vocabulary object from pickle file"""
    try:
        with open('vocabulary.pkl', 'rb') as f:
            vocab = pickle.load(f)
        st.success(f"✓ Loaded vocabulary: {len(vocab)} words")
        return vocab
    except FileNotFoundError:
        st.error("❌ vocabulary.pkl not found! Please upload it to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading vocabulary: {e}")
        st.stop()


@st.cache_resource
def load_models():
    """Load trained encoder and decoder models"""
    try:
        # Load checkpoint
        checkpoint = torch.load('caption_model.pth', map_location='cpu')
        
        # Reconstruct encoder
        encoder = ImageEncoder(encoded_image_size=7)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()
        
        # Reconstruct decoder using saved hyperparameters
        decoder = DecoderWithAttention(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            vocab_size=checkpoint['vocab_size'],
            encoder_dim=2048,
            attention_dim=512,
            num_layers=checkpoint.get('num_layers', 2),
            dropout=0.5
        )
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.eval()
        
        st.success("✓ Models loaded successfully")
        
        # Display model info
        with st.expander("📊 Model Information"):
            st.write(f"**Embedding Dimension:** {checkpoint['embedding_dim']}")
            st.write(f"**Hidden Dimension:** {checkpoint['hidden_dim']}")
            st.write(f"**Vocabulary Size:** {checkpoint['vocab_size']}")
            st.write(f"**Number of Layers:** {checkpoint.get('num_layers', 2)}")
            if 'epoch' in checkpoint:
                st.write(f"**Training Epochs:** {checkpoint['epoch']}")
        
        return encoder, decoder, checkpoint
        
    except FileNotFoundError:
        st.error("❌ caption_model.pth not found! Please upload it to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Make sure the model architecture matches your training code!")
        st.stop()


# ============================================================================
# Image Processing
# ============================================================================

def preprocess_image(image):
    """
    Transform image for model input
    Matches the ResNet50 preprocessing used in training
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# ============================================================================
# Caption Generation Functions
# ============================================================================

def generate_caption_greedy(encoder, decoder, image_tensor, vocab, max_length=50):
    """
    Generate caption using greedy search (picks most likely word at each step)
    Fast but may not produce the best caption
    """
    with torch.no_grad():
        # Extract features
        encoder_out = encoder(image_tensor)  # (1, 7, 7, 2048)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten encoder output
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, 49, 2048)
        
        # Initialize hidden state
        h, c = decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        current_word = torch.LongTensor([vocab.word2idx['<start>']])
        caption_indices = []
        
        for _ in range(max_length):
            # Embed current word
            embeddings = decoder.embedding(current_word).unsqueeze(1)  # (1, 1, embedding_dim)
            
            # Get attention
            hidden_state = h[-1]  # Last layer hidden state
            attention_weighted_encoding, _ = decoder.attention(encoder_out, hidden_state)
            
            # Concatenate embedding and attention
            lstm_input = torch.cat([embeddings.squeeze(1), attention_weighted_encoding], dim=1)
            lstm_input = lstm_input.unsqueeze(1)
            
            # LSTM step
            lstm_out, (h, c) = decoder.lstm(lstm_input, (h, c))
            
            # Predict next word
            scores = decoder.fc(lstm_out.squeeze(1))  # (1, vocab_size)
            predicted_idx = scores.argmax(dim=1).item()
            
            # Stop if <end> token
            if predicted_idx == vocab.word2idx['<end>']:
                break
            
            caption_indices.append(predicted_idx)
            current_word = torch.LongTensor([predicted_idx])
        
        # Decode to words
        caption = vocab.decode(caption_indices)
        return caption


def generate_caption_beam_search(encoder, decoder, image_tensor, vocab, beam_size=3, max_length=50):
    """
    Generate caption using beam search (explores multiple paths)
    Slower but generally produces better captions
    """
    with torch.no_grad():
        # Extract features
        encoder_out = encoder(image_tensor)  # (1, 7, 7, 2048)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten encoder output
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, 49, 2048)
        
        # Expand encoder output for beam search
        encoder_out = encoder_out.expand(beam_size, -1, -1)  # (beam_size, 49, 2048)
        
        # Initialize hidden state
        h, c = decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]] * beam_size)  # (beam_size, 1)
        seqs = k_prev_words  # (beam_size, 1)
        top_k_scores = torch.zeros(beam_size, 1)  # (beam_size, 1)
        
        complete_seqs = []
        complete_seqs_scores = []
        
        step = 1
        
        while True:
            # Embed previous words
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (beam_size, embedding_dim)
            
            # Get attention
            hidden_state = h[-1]  # (beam_size, hidden_dim)
            attention_weighted_encoding, _ = decoder.attention(encoder_out, hidden_state)
            
            # Concatenate embedding and attention
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            lstm_input = lstm_input.unsqueeze(1)  # (beam_size, 1, embedding_dim + encoder_dim)
            
            # LSTM step
            lstm_out, (h, c) = decoder.lstm(lstm_input, (h, c))
            
            # Predict next word scores
            scores = decoder.fc(lstm_out.squeeze(1))  # (beam_size, vocab_size)
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            
            # Add previous scores
            scores = top_k_scores.expand_as(scores) + scores  # (beam_size, vocab_size)
            
            # For first step, all k seqs will have same score
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
            else:
                # Find top k scores across all beams
                top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)
            
            # Convert to beam indices and word indices
            prev_word_inds = top_k_words // vocab_size  # Which beam it came from
            next_word_inds = top_k_words % vocab_size   # Which word it is
            
            # Build next step sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # Check which sequences are complete (hit <end> token)
            incomplete_inds = []
            for ind, next_word in enumerate(next_word_inds):
                if next_word == vocab.word2idx['<end>']:
                    complete_seqs.append(seqs[ind].tolist())
                    complete_seqs_scores.append(top_k_scores[ind].item())
                else:
                    incomplete_inds.append(ind)
            
            # Stop if max length reached or all beams complete
            if len(complete_seqs) >= beam_size or step >= max_length:
                break
            
            # Prepare for next iteration
            seqs = seqs[incomplete_inds]
            h = h[:, incomplete_inds, :]
            c = c[:, incomplete_inds, :]
            encoder_out = encoder_out[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
            step += 1
        
        # Select best sequence
        if complete_seqs:
            best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[best_seq_idx]
        else:
            # Fallback if no complete sequences
            best_seq = seqs[0].tolist()
        
        # Remove <start> and <end> tokens
        caption_indices = [idx for idx in best_seq 
                          if idx not in [vocab.word2idx['<start>'], 
                                        vocab.word2idx['<end>'], 
                                        vocab.word2idx['<pad>']]]
        
        # Decode to words
        caption = vocab.decode(caption_indices)
        return caption


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Title and description
    st.title("🖼️ Image Caption Generator")
    st.markdown("""
    Upload an image and the AI will generate a descriptive caption using a trained 
    **ResNet50 + 2-Layer LSTM** model with attention mechanism.
    """)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    generation_method = st.sidebar.selectbox(
        "Caption Generation Method",
        ["Beam Search (Better Quality)", "Greedy Search (Faster)"],
        help="Beam search explores multiple possibilities, greedy picks the most likely word at each step"
    )
    
    if "Beam" in generation_method:
        beam_size = st.sidebar.slider("Beam Size", 1, 5, 3, 
                                      help="Higher = better quality but slower")
    else:
        beam_size = 1
    
    max_length = st.sidebar.slider("Max Caption Length", 10, 100, 50)
    
    # Load models
    st.sidebar.header("📦 Model Status")
    with st.spinner("Loading models..."):
        vocab = load_vocabulary()
        encoder, decoder, checkpoint = load_models()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Show image info
            with st.expander("ℹ️ Image Information"):
                st.write(f"**Size:** {image.size}")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
    
    with col2:
        st.header("✨ Generated Caption")
        
        if uploaded_file is not None:
            # Generate button
            if st.button("🚀 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Generate caption
                    if "Beam" in generation_method:
                        caption = generate_caption_beam_search(
                            encoder, decoder, image_tensor, vocab, 
                            beam_size=beam_size, max_length=max_length
                        )
                    else:
                        caption = generate_caption_greedy(
                            encoder, decoder, image_tensor, vocab, 
                            max_length=max_length
                        )
                    
                    # Display result
                    st.success("✓ Caption generated successfully!")
                    st.markdown(f"### 💬 {caption.capitalize()}")
                    
                    # Additional info
                    with st.expander("📊 Caption Details"):
                        words = caption.split()
                        st.write(f"**Word Count:** {len(words)}")
                        st.write(f"**Method:** {generation_method}")
                        if "Beam" in generation_method:
                            st.write(f"**Beam Size:** {beam_size}")
        else:
            st.info("👆 Please upload an image to generate a caption")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Model Architecture:</strong> ResNet50 (Encoder) + 2-Layer LSTM with Attention (Decoder)</p>
        <p><strong>Training Dataset:</strong> Flickr30k</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()