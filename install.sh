#!/bin/bash

# Step 1: Build the release binary
echo "Building the release binary..."
cargo build --release

# Step 2: Create user's local bin directory if it doesn't exist
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

# Step 3: Install the binary in user's local bin
echo "Installing the binary as 'ast' in $INSTALL_DIR..."
cp target/release/ast "$INSTALL_DIR/ast"

# Step 4: Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "Adding $INSTALL_DIR to PATH in ~/.bashrc..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Please run 'source ~/.bashrc' or start a new terminal session"
fi

# Step 5: Verify the installation
if command -v ast &> /dev/null; then
    echo "Installation successful! You can now use the 'ast' command."
else
    echo "Please restart your terminal or run: source ~/.bashrc"
    echo "Then you can use the 'ast' command"
fi