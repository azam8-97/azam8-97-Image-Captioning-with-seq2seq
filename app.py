"""
Image Caption Generator - Streamlit Application
Loads a trained ResNet50 + 2-Layer LSTM model to generate captions for uploaded images
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


class Encoder(nn.Module):
    """Encoder: Projects 2048-dim image features to hidden_size."""
    def __init__(self, feature_size=2048, hidden_size=512):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(feature_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        hidden = self.dropout(features)
        hidden = self.fc(hidden)
        hidden = self.bn(hidden)
        hidden = self.relu(hidden)
        return hidden


class Decoder(nn.Module):
    """Decoder: 2-layer LSTM-based caption generator."""
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2-layer LSTM with dropout applied between layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions[:, :-1])
        batch_size = features.size(0)

        # Initialize 2 layers of h and c
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)

        # Inject features into the first layer
        h0[0] = features
        c0[0] = features

        lengths = [l - 1 for l in lengths]
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed, (h0, c0))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)

        return outputs

    def generate(self, features, max_length=25, method='greedy', beam_width=5):
        """Generate caption using greedy or beam search"""
        if method == 'greedy':
            return self._greedy_search(features, max_length)
        elif method == 'beam':
            return self._beam_search(features, max_length, beam_width)

    def _greedy_search(self, features, max_length):
        """Greedy search for caption generation"""
        generated_ids = []
        h = torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device)
        h[0] = features
        c[0] = features

        input_id = torch.LongTensor([1]).to(features.device)  # <start>

        for _ in range(max_length):
            embedded = self.embedding(input_id).unsqueeze(1)
            output, (h, c) = self.lstm(embedded, (h, c))
            output = self.fc(output.squeeze(1))
            predicted_id = output.argmax(1)

            generated_ids.append(predicted_id.item())
            if predicted_id.item() == 2:  # <end>
                break
            input_id = predicted_id

        return generated_ids

    def _beam_search(self, features, max_length, beam_width):
        """Beam search for caption generation"""
        h = torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size).to(features.device)
        h[0] = features
        c[0] = features

        beams = [([1], 0.0, h, c)]

        for _ in range(max_length):
            new_beams = []
            for seq, score, h_s, c_s in beams:
                if seq[-1] == 2:  # <end>
                    new_beams.append((seq, score, h_s, c_s))
                    continue

                input_id = torch.LongTensor([seq[-1]]).to(features.device)
                embedded = self.embedding(input_id).unsqueeze(1)
                output, (h_n, c_n) = self.lstm(embedded, (h_s, c_s))
                output = self.fc(output.squeeze(1))

                log_probs = torch.log_softmax(output, dim=1)
                top_probs, top_ids = log_probs.topk(beam_width)

                for prob, idx in zip(top_probs[0], top_ids[0]):
                    new_beams.append((seq + [idx.item()], score + prob.item(), h_n, c_n))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(s[-1] == 2 for s, _, _, _ in beams):
                break

        return beams[0][0][1:]  # Remove <start> token


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
        # Load vocabulary to get vocab_size
        with open('vocabulary.pkl', 'rb') as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab)

        # Load checkpoint
        checkpoint = torch.load('best_model (2).pth', map_location=torch.device('cpu'))

        # Get hyperparameters from checkpoint or use defaults from training
        embed_size = checkpoint.get('embed_size', 256)
        hidden_size = checkpoint.get('hidden_size', 512)
        num_layers = checkpoint.get('num_layers', 2)

        # Reconstruct encoder
        encoder = Encoder(feature_size=2048, hidden_size=hidden_size)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()

        # Reconstruct decoder
        decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3
        )
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.eval()

        st.success("✓ Models loaded successfully")

        # Display model info
        with st.expander("📊 Model Information"):
            st.write(f"**Embedding Dimension:** {embed_size}")
            st.write(f"**Hidden Dimension:** {hidden_size}")
            st.write(f"**Vocabulary Size:** {vocab_size}")
            st.write(f"**Number of Layers:** {num_layers}")
            if 'epoch' in checkpoint:
                st.write(f"**Training Epochs:** {checkpoint['epoch']}")

        return encoder, decoder, checkpoint

    except FileNotFoundError:
        st.error("❌ best_model (2).pth not found! Please upload it to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Make sure the model architecture matches your training code!")
        st.stop()


# ============================================================================
# Feature Extraction
# ============================================================================

@st.cache_resource
def load_resnet():
    """Load ResNet50 model for feature extraction"""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet


def preprocess_image(image):
    """Transform image for ResNet50 input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def extract_features(image_tensor, resnet_model):
    """Extract features from image using ResNet50"""
    with torch.no_grad():
        features = resnet_model(image_tensor)
        features = features.view(features.size(0), -1)  # Flatten to (batch, 2048)
    return features


# ============================================================================
# Caption Generation Functions
# ============================================================================

def generate_caption(encoder, decoder, image_tensor, vocab, resnet_model,
                    method='greedy', max_length=50, beam_size=3):
    """
    Generate caption for an image

    Args:
        encoder: Encoder model
        decoder: Decoder model
        image_tensor: Preprocessed image tensor
        vocab: Vocabulary object
        resnet_model: ResNet50 for feature extraction
        method: 'greedy' or 'beam'
        max_length: Maximum caption length
        beam_size: Beam width for beam search

    Returns:
        Generated caption string
    """
    with torch.no_grad():
        # Extract features from image
        image_features = extract_features(image_tensor, resnet_model)

        # Encode features
        encoded_features = encoder(image_features)

        # Generate caption
        if method == 'beam':
            caption_indices = decoder.generate(
                encoded_features,
                max_length=max_length,
                method='beam',
                beam_width=beam_size
            )
        else:
            caption_indices = decoder.generate(
                encoded_features,
                max_length=max_length,
                method='greedy'
            )

        # Remove special tokens (<end>, <pad>)
        caption_indices = [idx for idx in caption_indices if idx not in [0, 2]]

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
    **ResNet50 + 2-Layer LSTM** model.
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

    max_length = st.sidebar.slider("Max Caption Length", 10, 100, 25)

    # Load models
    st.sidebar.header("📦 Model Status")
    with st.spinner("Loading models..."):
        vocab = load_vocabulary()
        encoder, decoder, checkpoint = load_models()
        resnet_model = load_resnet()

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
                    method = 'beam' if "Beam" in generation_method else 'greedy'
                    caption = generate_caption(
                        encoder, decoder, image_tensor, vocab, resnet_model,
                        method=method, max_length=max_length, beam_size=beam_size
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
        <p><strong>Model Architecture:</strong> ResNet50 (Feature Extractor) + Encoder + 2-Layer LSTM (Decoder)</p>
        <p><strong>Training Dataset:</strong> Flickr30k</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
