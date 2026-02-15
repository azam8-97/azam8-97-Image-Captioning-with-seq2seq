"""
Pre-Deployment Verification Script
Run this BEFORE deploying to Streamlit to catch issues early

Usage: python verify_deployment.py
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_file_exists(filename, required=True):
    """Check if a file exists and print its size"""
    exists = os.path.exists(filename)
    
    if exists:
        size_mb = os.path.getsize(filename) / (1024**2)
        print(f"✓ {filename:<30} ({size_mb:.2f} MB)")
        return True
    else:
        status = "REQUIRED" if required else "OPTIONAL"
        print(f"✗ {filename:<30} MISSING! [{status}]")
        return not required

def verify_vocabulary():
    """Verify vocabulary.pkl can be loaded"""
    print_header("STEP 1: Verifying Vocabulary")
    
    if not check_file_exists('vocabulary.pkl', required=True):
        return False
    
    try:
        import pickle
        
        with open('vocabulary.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        # Check if it's the Vocabulary class or just dictionaries
        if hasattr(vocab, 'word2idx'):
            # It's a Vocabulary object
            vocab_size = len(vocab.word2idx)
            print(f"\n✓ Loaded Vocabulary object")
            print(f"  - Vocabulary size: {vocab_size}")
            print(f"  - Has word2idx: {hasattr(vocab, 'word2idx')}")
            print(f"  - Has idx2word: {hasattr(vocab, 'idx2word')}")
            print(f"  - Has encode method: {hasattr(vocab, 'encode')}")
            print(f"  - Has decode method: {hasattr(vocab, 'decode')}")
            
            # Check special tokens
            special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
            print(f"\n  Special tokens:")
            for token in special_tokens:
                if token in vocab.word2idx:
                    print(f"    ✓ {token}: index {vocab.word2idx[token]}")
                else:
                    print(f"    ✗ {token}: MISSING!")
                    return False
            
            # Test encode/decode
            test_caption = "a dog running in the park"
            encoded = vocab.encode(test_caption)
            decoded = vocab.decode(encoded)
            print(f"\n  Encode/Decode Test:")
            print(f"    Original: {test_caption}")
            print(f"    Encoded:  {encoded}")
            print(f"    Decoded:  {decoded}")
            
        else:
            # It's just dictionaries
            print(f"\n✓ Loaded vocabulary dictionaries")
            if 'word2idx' in vocab:
                print(f"  - Vocabulary size: {len(vocab['word2idx'])}")
            else:
                print(f"  ✗ Missing 'word2idx' key!")
                return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading vocabulary: {e}")
        return False

def verify_model():
    """Verify caption_model.pth can be loaded"""
    print_header("STEP 2: Verifying Model Checkpoint")
    
    if not check_file_exists('caption_model.pth', required=True):
        return False
    
    try:
        import torch
        
        checkpoint = torch.load('caption_model.pth', map_location='cpu')
        
        print(f"\n✓ Loaded model checkpoint")
        print(f"\nCheckpoint contents:")
        for key in checkpoint.keys():
            if 'state_dict' in key:
                print(f"  ✓ {key}")
            else:
                print(f"  - {key}: {checkpoint[key]}")
        
        # Check required keys
        required_keys = ['encoder_state_dict', 'decoder_state_dict', 
                        'embedding_dim', 'hidden_dim', 'vocab_size']
        
        print(f"\nRequired hyperparameters:")
        all_present = True
        for key in required_keys:
            if key in checkpoint:
                if 'state_dict' not in key:
                    print(f"  ✓ {key}: {checkpoint[key]}")
                else:
                    print(f"  ✓ {key}: present")
            else:
                print(f"  ✗ {key}: MISSING!")
                all_present = False
        
        if not all_present:
            print("\n⚠ WARNING: Some required keys are missing!")
            print("  Your model may not load correctly in Streamlit.")
            return False
        
        # Check state dict sizes
        print(f"\nModel architecture info:")
        encoder_params = sum(p.numel() for p in checkpoint['encoder_state_dict'].values())
        decoder_params = sum(p.numel() for p in checkpoint['decoder_state_dict'].values())
        total_params = encoder_params + decoder_params
        
        print(f"  - Encoder parameters: {encoder_params:,}")
        print(f"  - Decoder parameters: {decoder_params:,}")
        print(f"  - Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return False

def verify_consistency():
    """Verify vocabulary and model are consistent"""
    print_header("STEP 3: Verifying Vocabulary-Model Consistency")
    
    try:
        import pickle
        import torch
        
        # Load vocabulary
        with open('vocabulary.pkl', 'rb') as f:
            vocab = pickle.load(f)
        
        # Load model
        checkpoint = torch.load('caption_model.pth', map_location='cpu')
        
        # Get vocab sizes
        if hasattr(vocab, 'word2idx'):
            vocab_size_from_vocab = len(vocab.word2idx)
        else:
            vocab_size_from_vocab = len(vocab['word2idx'])
        
        vocab_size_from_model = checkpoint['vocab_size']
        
        print(f"Vocabulary size comparison:")
        print(f"  - From vocabulary.pkl: {vocab_size_from_vocab}")
        print(f"  - From caption_model.pth: {vocab_size_from_model}")
        
        if vocab_size_from_vocab == vocab_size_from_model:
            print(f"\n✓ Vocabulary sizes match!")
            return True
        else:
            print(f"\n✗ MISMATCH! Vocabulary sizes don't match!")
            print(f"  This means your vocabulary and model are from different training runs.")
            print(f"  You MUST re-save both from the SAME training run!")
            return False
        
    except Exception as e:
        print(f"\n✗ Error checking consistency: {e}")
        return False

def verify_dependencies():
    """Check if required Python packages are installed"""
    print_header("STEP 4: Verifying Python Dependencies")
    
    required_packages = {
        'streamlit': '1.31.0',
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'PIL': '10.0.0',
        'numpy': '1.24.0'
    }
    
    all_installed = True
    
    for package, min_version in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
                pkg_name = 'Pillow'
            else:
                module = __import__(package)
                version = module.__version__
                pkg_name = package
            
            print(f"✓ {pkg_name:<15} version {version}")
            
        except ImportError:
            print(f"✗ {package:<15} NOT INSTALLED!")
            all_installed = False
    
    if not all_installed:
        print(f"\n⚠ Some packages are missing!")
        print(f"  Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_app_file():
    """Check if app.py exists"""
    print_header("STEP 5: Verifying Streamlit App")
    
    if not check_file_exists('app.py', required=True):
        print("\n✗ app.py not found!")
        print("  Make sure you have the Streamlit app file in this directory.")
        return False
    
    # Check if it contains required components
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        required_components = [
            'class Vocabulary',
            'class ImageEncoder',
            'class DecoderWithAttention',
            'def load_vocabulary',
            'def load_models',
            'def generate_caption'
        ]
        
        print(f"\nChecking app.py structure:")
        all_present = True
        for component in required_components:
            if component in content:
                print(f"  ✓ {component}")
            else:
                print(f"  ✗ {component} - MISSING!")
                all_present = False
        
        if not all_present:
            print(f"\n⚠ Some components are missing from app.py!")
            print(f"  Make sure you copied the complete code.")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error reading app.py: {e}")
        return False

def main():
    """Run all verification checks"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Image Caption Generator - Deployment Verifier       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("This script will verify that your deployment is ready.")
    print("It will check:")
    print("  1. Vocabulary file exists and can be loaded")
    print("  2. Model checkpoint exists and can be loaded")
    print("  3. Vocabulary and model are consistent")
    print("  4. Required Python packages are installed")
    print("  5. Streamlit app file is present and complete")
    
    # Run all checks
    results = []
    
    results.append(("Vocabulary", verify_vocabulary()))
    results.append(("Model Checkpoint", verify_model()))
    results.append(("Consistency", verify_consistency()))
    results.append(("Dependencies", verify_dependencies()))
    results.append(("App File", check_app_file()))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║               ✓ ALL CHECKS PASSED!                       ║
║                                                          ║
║  Your deployment is ready! You can now run:             ║
║                                                          ║
║      streamlit run app.py                                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """)
        return 0
    else:
        print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║               ✗ SOME CHECKS FAILED                       ║
║                                                          ║
║  Please fix the issues above before deploying.          ║
║  Check the error messages for specific problems.        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """)
        return 1

if __name__ == "__main__":
    sys.exit(main())