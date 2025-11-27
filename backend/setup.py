"""Setup script - ENHANCED with validation."""
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(text):
    """Print step info."""
    print(f"\n→ {text}")

def print_success(text):
    """Print success message."""
    print(f"✓ {text}")

def print_error(text):
    """Print error message."""
    print(f"✗ {text}")

def install_dependencies():
    """Install Python dependencies."""
    print_step("Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
        ])
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download spaCy English model."""
    print_step("Downloading spaCy English model...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print_success("spaCy model downloaded")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to download spaCy model")
        print("  You can install it manually: python -m spacy download en_core_web_sm")
        return False

def download_vosk_model():
    """Download Vosk model automatically."""
    print_step("Checking Vosk model...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_dir = models_dir / "vosk-model-en-us-0.22"
    
    if model_dir.exists() and (model_dir / "am" / "final.mdl").exists():
        print_success("Vosk model already exists")
        return True
    
    print_step("Downloading Vosk model (this may take a few minutes)...")
    
    # Model URL
    model_url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
    zip_path = models_dir / "vosk-model.zip"
    
    try:
        # Download with progress
        print(f"  Downloading from: {model_url}")
        urllib.request.urlretrieve(model_url, zip_path)
        print_success("Download complete")
        
        # Extract
        print_step("Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Clean up
        zip_path.unlink()
        
        print_success("Vosk model installed")
        return True
        
    except Exception as e:
        print_error(f"Failed to download Vosk model: {e}")
        print("\nManual installation:")
        print("  1. Visit: https://alphacephei.com/vosk/models")
        print("  2. Download: vosk-model-en-us-0.22")
        print("  3. Extract to: backend/models/vosk-model-en-us-0.22")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print_step("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print_success("NLTK data downloaded")
        return True
    except Exception as e:
        print_error(f"Failed to download NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print_step("Creating directories...")
    dirs = ["models", "logs", "temp"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print_success("Directories created")
    return True

def create_env_file():
    """Create .env file from example."""
    print_step("Setting up environment file...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print_success(".env file created")
    else:
        # Create basic .env
        with open(env_file, 'w') as f:
            f.write("# Environment Configuration\n")
            f.write("VOSK_MODEL_PATH=models/vosk-model-en-us-0.22\n")
            f.write("GRAMMAR_MODEL_NAME=t5-small\n")
            f.write("TARGET_LATENCY_MS=1500\n")
            f.write("USE_T5_MODEL=False\n")
        print_success(".env file created with defaults")
    
    return True

def verify_installation():
    """Verify installation."""
    print_header("VERIFYING INSTALLATION")
    
    all_good = True
    
    # Check Vosk model
    print_step("Checking Vosk model...")
    model_path = Path("models/vosk-model-en-us-0.22/am/final.mdl")
    if model_path.exists():
        print_success("Vosk model: OK")
    else:
        print_error("Vosk model: NOT FOUND")
        all_good = False
    
    # Check spaCy model
    print_step("Checking spaCy model...")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print_success("spaCy model: OK")
    except:
        print_error("spaCy model: NOT FOUND")
        all_good = False
    
    # Check key packages
    print_step("Checking key packages...")
    packages = ['fastapi', 'vosk', 'transformers', 'torch']
    for package in packages:
        try:
            __import__(package)
            print_success(f"{package}: OK")
        except ImportError:
            print_error(f"{package}: NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Run setup."""
    print_header("INTELLIGENT SPEECH DICTATION ENGINE - SETUP")
    print("This script will set up the backend environment")
    
    os.chdir(Path(__file__).parent)
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading spaCy model", download_spacy_model),
        ("Downloading Vosk model", download_vosk_model),
        ("Downloading NLTK data", download_nltk_data),
        ("Creating environment file", create_env_file),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        if step_func():
            success_count += 1
    
    print_header("SETUP SUMMARY")
    print(f"Completed {success_count}/{len(steps)} steps")
    
    if success_count == len(steps):
        # Verify installation
        if verify_installation():
            print_header("SETUP COMPLETE ✓")
            print("\nNext steps:")
            print("  1. Review .env file and adjust settings if needed")
            print("  2. Run: python main.py")
            print("  3. Visit: http://localhost:8000/docs")
        else:
            print_header("SETUP COMPLETE WITH WARNINGS ⚠")
            print("\nSome components need attention. See errors above.")
    else:
        print_header("SETUP INCOMPLETE ✗")
        print("\nSome steps failed. Please check errors above and retry.")
        print("You can also install components manually.")

if __name__ == "__main__":
    main()