from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap
import io
import base64
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add Stability AI API as an alternative
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')

def generate_with_starryai(api_key, prompt, width=512, height=512):
    """Generate an image using the StarryAI API"""
    # This code remains as a fallback, but we'll prioritize Stability AI
    # ...existing StarryAI implementation...
    
def generate_with_stability(api_key, prompt, width=512, height=512):
    """Generate an image using the Stability AI API (DreamStudio)"""
    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    
    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "image/*"
    }
    
    data = {
        "prompt": prompt,
        "output_format": "webp",
        "width": str(width),
        "height": str(height)
    }
    
    try:
        print(f"Using Stability AI v2beta API to generate image with prompt: {prompt}")
        response = requests.post(
            url,
            headers=headers,
            files={"none": ''},
            data=data
        )
        
        print(f"Stability API response status: {response.status_code}")
        
        if response.status_code != 200:
            try:
                error_info = response.json()
                print(f"Error from Stability API: {error_info}")
            except:
                print(f"Error from Stability API: {response.text}")
            return None
            
        # The response directly contains the image data
        image = Image.open(io.BytesIO(response.content))
        return image
        
    except Exception as e:
        print(f"Error generating image with Stability AI: {str(e)}")
        return None

def generate_placeholder_image(prompt, width=512, height=512):
    """Generate a placeholder image with the prompt text (no API key required)"""
    # Create a blank image with a random background color
    bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a font (you may need to adjust the path)
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font if arial.ttf is not available
        font = ImageFont.load_default()
    
    # Add text to the image
    text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    text = f"Prompt: {prompt}\n(Placeholder Image)"
    
    # Wrap text
    lines = textwrap.wrap(text, width=30)
    y_text = height // 4
    for line in lines:
        # Use font.getbbox() instead of draw.textsize() which is deprecated
        bbox = font.getbbox(line)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - text_width) / 2, y_text), line, font=font, fill=text_color)
        y_text += text_height + 10
    
    # Add some random shapes for visual interest
    for _ in range(5):
        shape_type = random.choice(['ellipse', 'rectangle'])
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        # Ensure the coordinates are properly ordered (x1,y1 is top-left, x2,y2 is bottom-right)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        if shape_type == 'ellipse':
            draw.ellipse([left, top, right, bottom], outline=color, width=3)
        else:
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
    
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt', '')
    width = int(request.form.get('width', 512))
    height = int(request.form.get('height', 512))
    
    # Try Stability API first, then StarryAI as backup, then placeholder as final fallback
    stability_key = os.getenv('STABILITY_API_KEY')
    starry_key = os.getenv('STARRYAI_API_KEY')
    
    image = None
    
    # Try Stability AI first (it's more reliable)
    if stability_key:
        print(f"Using Stability AI API to generate image with prompt: {prompt}")
        image = generate_with_stability(stability_key, prompt, width, height)
    
    # Fall back to StarryAI if Stability fails or isn't configured
    if image is None and starry_key and starry_key != 'your_api_key_here':
        print(f"Stability AI failed or not configured. Trying StarryAI with prompt: {prompt}")
        image = generate_with_starryai(starry_key, prompt, width, height)
    
    # Fall back to placeholder as last resort
    if image is None:
        print("All API calls failed, using placeholder image instead")
        image = generate_placeholder_image(prompt, width, height)
    
    # Save the image to a file
    filename = f"generated_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    
    # Pass image_url and remove the 'hidden' class on the result div
    return render_template('index.html', 
                          image_url=f"/static/generated/{filename}",
                          show_result=True)

if __name__ == '__main__':
    app.run(debug=True)
