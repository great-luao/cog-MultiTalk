#!/usr/bin/env python3
"""
Persistent MultiTalk model server

Loads all models once and serves prediction requests via HTTP.
Run this in a separate terminal (e.g., tmux) to avoid reloading models for each test.
"""

import os
import sys
import argparse
from typing import Optional
from flask import Flask, request, jsonify
# Reuse the predictor implementation from run_predict
from run_predict import MultiTalkPredictor

# Ensure project path is on PYTHONPATH
PROJECT_ROOTS = [
    "/workspace/cog-MultiTalk",  # container path (default)
    os.path.dirname(os.path.abspath(__file__)),  # local path
]
for root in PROJECT_ROOTS:
    if root not in sys.path and os.path.exists(root):
        sys.path.insert(0, root)

# Model cache envs
MODEL_CACHE_DEFAULTS = [
    "/workspace/cog-MultiTalk/weights",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
]
MODEL_CACHE = next((p for p in MODEL_CACHE_DEFAULTS if os.path.exists(os.path.dirname(p) if p.endswith('.tar') else p)), MODEL_CACHE_DEFAULTS[0])
os.environ.setdefault("HF_HOME", MODEL_CACHE)
os.environ.setdefault("TORCH_HOME", MODEL_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", MODEL_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_CACHE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", MODEL_CACHE)


def create_app() -> Flask:
    app = Flask(__name__)

    # Instantiate predictor once; models stay in GPU/CPU memory
    predictor = MultiTalkPredictor()

    @app.get("/health")
    def health() -> tuple[dict, int]:
        return {"status": "ok"}, 200

    @app.post("/predict")
    def predict_route():
        try:
            data = request.get_json(force=True, silent=False) or {}

            image: str = data.get("image")
            first_audio: str = data.get("first_audio")
            prompt: str = data.get("prompt", "A smiling man and woman wearing headphones sit in front of microphones, appearing to host a podcast.")
            second_audio: Optional[str] = data.get("second_audio")
            num_frames: int = int(data.get("num_frames", 81))
            sampling_steps: int = int(data.get("sampling_steps", 40))
            seed: Optional[int] = data.get("seed")
            turbo: bool = bool(data.get("turbo", False))
            output_path: Optional[str] = data.get("output_path")
            audio_type: Optional[str] = data.get("audio_type")
            bbox = data.get("bbox")

            if not image or not first_audio:
                return jsonify({
                    "error": "Both 'image' and 'first_audio' are required."
                }), 400

            result_path = predictor.predict(
                image=image,
                first_audio=first_audio,
                prompt=prompt,
                second_audio=second_audio,
                num_frames=num_frames,
                sampling_steps=sampling_steps,
                seed=seed,
                turbo=turbo,
                output_path=output_path,
                audio_type=audio_type,
                bbox=bbox,
            )

            return jsonify({"output_path": result_path}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Start MultiTalk model server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    app = create_app()
    # Single-threaded to avoid concurrent GPU races while testing
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()


