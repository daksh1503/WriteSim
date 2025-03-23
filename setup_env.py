#!/usr/bin/env python3
"""
Setup script for WriteSim project.
Installs dependencies in the correct order with proper error handling.
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, description=None):
    """Run a command and print the output."""
    if description:
        print(f"\n{'-' * 80}\n{description}\n{'-' * 80}")
    
    print(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def check_system():
    """Check system requirements and print information."""
    print(f"\nSystem information:")
    print(f"  - OS: {platform.system()} {platform.release()}")
    print(f"  - Python: {platform.python_version()}")
    
    # Check if running on macOS
    if platform.system() == "Darwin":
        print(f"  - macOS version: {platform.mac_ver()[0]}")
        
        # Check for Apple Silicon
        if platform.machine() == "arm64":
            print("  - Apple Silicon (M1/M2) detected")
            print("  - Will configure PyTorch for Metal acceleration")
        else:
            print("  - Intel Mac detected")

def setup_virtual_env():
    """Set up a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("\nVirtual environment already exists.")
        return True
    
    print("\nCreating virtual environment...")
    if not run_command([sys.executable, "-m", "venv", "venv"]):
        print("Failed to create virtual environment. Please install venv package.")
        return False
    
    return True

def install_torch():
    """Install PyTorch separately with the correct configuration."""
    # Determine the correct PyTorch installation command based on platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon - use MPS
        cmd = [
            "venv/bin/pip", "install", 
            "torch>=2.0.0", 
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"
        ]
    else:
        # Default installation
        cmd = ["venv/bin/pip", "install", "torch>=2.0.0"]
    
    return run_command(cmd, "Installing PyTorch")

def install_binary_packages():
    """Install binary packages that don't require compilation."""
    packages = [
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "protobuf>=3.20.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "gradio>=3.50.0",
        "tensorboard>=2.15.0",
        "accelerate>=0.20.0"
    ]
    
    cmd = ["venv/bin/pip", "install"] + packages
    return run_command(cmd, "Installing main packages")

def install_peft():
    """Install PEFT separately."""
    return run_command(
        ["venv/bin/pip", "install", "peft>=0.5.0"],
        "Installing PEFT"
    )

def install_sentencepiece():
    """Install sentencepiece with special handling."""
    # First try to install the wheel directly
    if run_command(
        ["venv/bin/pip", "install", "sentencepiece>=0.1.99", "--no-build-isolation"],
        "Installing sentencepiece (attempt 1 - no build isolation)"
    ):
        return True
    
    # If that fails, try with the --no-binary option to force compilation
    print("\nFirst attempt failed, trying alternative approach...")
    
    # On macOS, we need pkg-config and cmake
    if platform.system() == "Darwin":
        print("\nYou may need to install build dependencies with Homebrew:")
        print("  brew install pkg-config cmake")
        print("Please run this command and then re-run this script.")
        
        response = input("\nWould you like to continue trying to install anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return run_command(
        ["venv/bin/pip", "install", "sentencepiece>=0.1.99", "--no-binary", "sentencepiece"],
        "Installing sentencepiece (attempt 2 - force compilation)"
    )

def main():
    """Main setup function."""
    print("\n" + "=" * 80)
    print("WriteSim Setup Assistant".center(80))
    print("=" * 80)
    
    # Check system information
    check_system()
    
    # Set up virtual environment
    if not setup_virtual_env():
        return False
    
    # Update pip and setuptools
    if not run_command(["venv/bin/pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                     "Updating pip, setuptools, and wheel"):
        return False
    
    # Install PyTorch
    if not install_torch():
        return False
    
    # Install binary packages
    if not install_binary_packages():
        return False
    
    # Install PEFT
    if not install_peft():
        return False
    
    # Install sentencepiece
    if not install_sentencepiece():
        print("\nWARNING: Failed to install sentencepiece. Some functionality may be limited.")
        print("You may need to install development tools and try again:")
        print("  brew install pkg-config cmake")
    
    print("\n" + "=" * 80)
    print("Setup completed!".center(80))
    print("=" * 80)
    print("\nTo activate the virtual environment, run:")
    print("  source venv/bin/activate  # On Unix/macOS")
    print("  venv\\Scripts\\activate     # On Windows")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 