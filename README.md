# Text To Image Generator

A web application that generates high-quality images from text prompts using Stable Diffusion v1.5.

## Features

- Text-to-image generation using Stable Diffusion v1.5
- Real-time loading progress indicator with timer
- Supports both GPU and CPU processing
- Optimized for memory efficiency and speed
- High-quality image output with enhanced clarity

## Requirements

- Python 3.8+
- Flask
- PyTorch
- Diffusers
- Hugging Face account with API token

## Setup

1. Clone this repository
```bash
git clone https://github.com/your-username/stable-diffusion-webapp.git
cd stable-diffusion-webapp
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face token
```bash
# Copy the example file
cp .env.example .env

# Edit the .env file and add your token
# HUGGINGFACE_TOKEN=your_token_here
```

4. Run the application
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter a descriptive text prompt in the input field
2. Click "Generate Image"
3. Wait for the image to be generated (progress will be displayed)
4. View and download your generated image

## Performance Notes

- GPU mode: Faster generation with higher quality output
- CPU mode: Slower but works on machines without dedicated GPU
- Processing time varies based on the complexity of your prompt

## License

MIT License

## Acknowledgments

- [Stable Diffusion by Stability AI](https://stability.ai/stable-diffusion)
- [Hugging Face Diffusers library](https://github.com/huggingface/diffusers)
