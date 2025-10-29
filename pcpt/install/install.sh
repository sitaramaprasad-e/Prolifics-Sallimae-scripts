#!/usr/bin/env bash
set -e

echo "============================================================"
echo " 🚀 Starting PCPT Installation"
echo "============================================================"

uname -m
echo "🔍 Detected system architecture: $(uname -m)"

# Ask for full image:tag and default to greghodgkinson/pcpt:edge
read -p "Enter the PCPT image (name:tag) [default: greghodgkinson/pcpt:edge]: " PCPT_IMAGE
PCPT_IMAGE=${PCPT_IMAGE:-greghodgkinson/pcpt:edge}

# Escape / and & for sed safety
ESCAPED_IMAGE=$(printf '%s' "$PCPT_IMAGE" | sed 's/[\/&]/\\&/g')


# Pull the selected image
podman pull "docker.io/$PCPT_IMAGE"
echo "✅ PCPT container image pulled: $PCPT_IMAGE"

sudo cp install/pcpt.sh /usr/local/bin
# Update the installed copy of pcpt.sh with the selected image
sudo sed -i.bak \
  -e "s|greghodgkinson/pcpt:edge|${ESCAPED_IMAGE}|g" \
  -e "s|greghodgkinson/pcpt:latest|${ESCAPED_IMAGE}|g" \
  /usr/local/bin/pcpt.sh 2>/dev/null || true
sudo chmod +x /usr/local/bin/pcpt.sh
echo "✅ PCPT client script deployed to /usr/local/bin"

mkdir -p ~/.pcpt
mkdir -p ~/.pcpt/config
mkdir -p ~/.pcpt/log
cp -rf install/.pcpt/hints/* ~/.pcpt/hints
cp -rf install/.pcpt/prompts/* ~/.pcpt/prompts
if [ ! -f ~/.pcpt/config/pcpt.config ]; then
    cp -rf install/.pcpt/config/* ~/.pcpt/config
fi
echo "✅ PCPT install files and configuration copied to ~/.pcpt"

# If running under WSL on Windows, ensure /usr/local/bin is in PATH (now and later)
if grep -qi microsoft /proc/version 2>/dev/null; then
    if ! grep -q '/usr/local/bin' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
        echo "🔧 Added /usr/local/bin to PATH in ~/.bashrc (WSL detected)"
    fi
    export PATH="/usr/local/bin:$PATH"
    echo "🔧 Updated PATH for current WSL session"
fi

echo "============================================================"
echo " 🎉 PCPT installation completed successfully!"
echo " 👉 You can now run 'pcpt.sh' from any terminal."
echo "============================================================"

read -p "Press Enter to exit..."
