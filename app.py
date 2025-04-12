import os
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure requests with retry mechanism
def get_session_with_retries():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Increase timeout for large downloads (10 minutes)
    session.request = lambda method, url, **kwargs: super(requests.Session, session).request(
        method=method, url=url, timeout=600, **kwargs
    )
    return session

# Get API token from .env
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face token not found in .env file")

# Set device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Initialize the pipeline with retry mechanism
def load_model_with_retries(max_attempts=3, delay=5):
    """Load the model with retry mechanism for handling connection issues"""
    for attempt in range(max_attempts):
        try:
            print(f"Loading model, attempt {attempt+1}/{max_attempts}...")
            
            # Create a custom session with retries for the huggingface_hub
            session = get_session_with_retries()
            
            # Initialize the pipeline with the custom session
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",  # Using v1.5 for better quality
                use_auth_token=hf_token,
                safety_checker=None,  # Disabled for simplicity
                torch_dtype=torch_dtype,
                custom_pipeline="lpw_stable_diffusion",  # Low memory pipeline option
            )
            
            # Enable memory efficient attention if using CUDA
            if device == "cuda":
                pipe.enable_attention_slicing()
                # Enable sequential CPU offload to save VRAM
                pipe.enable_sequential_cpu_offload()
            else:
                pipe = pipe.to(device)
                
            print("Model loaded successfully!")
            return pipe
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (attempt + 1)
                print(f"Connection error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to load model after {max_attempts} attempts. Error: {str(e)}")
                raise

# Try to load the model with retries
try:
    pipe = load_model_with_retries()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # We'll handle this in the routes

@app.route("/", methods=["GET", "POST"])
def home():
    # Pass device information to template
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    return render_template("index.html", device=device_info)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/generate", methods=["POST"])
def generate_image():
    prompt = request.form.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Improved generation parameters
        # Fewer steps if on CPU for speed, more on GPU for quality
        num_steps = 30 if device == "cuda" else 15
        
        # Generate higher guidance scale for clearer images
        image = pipe(
            prompt, 
            num_inference_steps=num_steps,
            guidance_scale=7.5,  # Higher value = more adherence to prompt (clearer results)
            width=512,
            height=512,
        ).images[0]
        
        # Enhance image quality by saving with higher quality settings
        img_io = BytesIO()
        image.save(img_io, "PNG", quality=95)
        img_io.seek(0)
        
        return send_file(img_io, mimetype="image/png", download_name="generated_image.png")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)