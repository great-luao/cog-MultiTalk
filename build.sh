#!/bin/bash
# 构建MultiTalk Docker镜像

echo "🔨 Building MultiTalk Runtime Docker image..."
docker build -t multitalk-runtime:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Next steps:"
    echo "1. Push to Docker Hub (optional):"
    echo "   docker tag multitalk-runtime:latest <your-username>/multitalk-runtime:latest"
    echo "   docker push <your-username>/multitalk-runtime:latest"
    echo ""
    echo "2. Use in RunPod with this image name"
else
    echo "❌ Build failed!"
    exit 1
fi
