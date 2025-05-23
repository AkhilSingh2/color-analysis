import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dlib
import pandas as pd
from collections import Counter
from PIL import Image, ImageColor
import colorsys
import base64
import io
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

# Create the static directory for Swagger JSON file
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Create Swagger JSON file
swagger_json = {
    "swagger": "2.0",
    "info": {
        "title": "Color Analysis API",
        "description": "API for analyzing face images to determine skin undertone, contrast, and recommend complementary colors",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["http"],
    "paths": {
        "/analyze": {
            "post": {
                "summary": "Analyze face image and recommend colors",
                "description": "Analyzes a face image to determine skin undertone, contrast level, extract dominant skin colors, and recommend complementary colors",
                "produces": ["application/json"],
                "consumes": ["application/json"],
                "parameters": [
                    {
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "description": "Image data in base64 format",
                        "schema": {
                            "type": "object",
                            "required": ["image"],
                            "properties": {
                                "image": {
                                    "type": "string",
                                    "description": "Base64 encoded image data"
                                }
                            }
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful analysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "undertone": {
                                    "type": "string",
                                    "description": "Skin undertone (Warm/Cool)"
                                },
                                "contrast": {
                                    "type": "string",
                                    "description": "Skin contrast level (Low/Medium/High)"
                                },
                                "skin_colors": {
                                    "type": "array",
                                    "description": "Dominant skin colors",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "rgb": {
                                                "type": "array",
                                                "items": {"type": "integer"}
                                            },
                                            "hex": {"type": "string"}
                                        }
                                    }
                                },
                                "recommended_colors": {
                                    "type": "array",
                                    "description": "Recommended complementary colors",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "rgb": {
                                                "type": "array",
                                                "items": {"type": "integer"}
                                            },
                                            "hex": {"type": "string"},
                                            "explanation": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Bad request",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"}
                            }
                        }
                    },
                    "500": {
                        "description": "Internal server error",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "summary": "Health check",
                "description": "Returns the health status of the API",
                "produces": ["application/json"],
                "responses": {
                    "200": {
                        "description": "Health status",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
}

# Write Swagger JSON to file
with open(os.path.join(os.path.dirname(__file__), 'static', 'swagger.json'), 'w') as f:
    json.dump(swagger_json, f)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup Swagger UI
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Color Analysis API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Path to the shape predictor model - you need to download this separately
PREDICTOR_PATH = os.environ.get('PREDICTOR_PATH', './shape_predictor_68_face_landmarks.dat')

# Function to convert image from base64 to CV2 format
def base64_to_cv2(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# Function to convert RGB to hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# Function to detect faces in an image
def detect_face(img):
    # Convert to RGB (dlib works better with RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load the face detector from dlib
    detector = dlib.get_frontal_face_detector()
    
    # Load the facial landmark predictor
    try:
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except RuntimeError as e:
        return None, None, None, str(e)
    
    # Detect faces
    faces = detector(rgb_img)
    
    if len(faces) == 0:
        return None, None, img, "No face detected in the image."
    
    # Get the largest face (assuming the main subject)
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Get facial landmarks
    landmarks = predictor(rgb_img, largest_face)
    
    # Create a mask for the face
    mask = np.zeros_like(rgb_img[:,:,0])
    
    # Fill in the face region
    face_points = []
    for i in range(0, 68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        face_points.append((x, y))
    
    face_points = np.array(face_points, dtype=np.int32)
    cv2.fillPoly(mask, [face_points], 255)
    
    # Create a masked image containing only face pixels
    masked_face = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    
    # Extract face from the image
    x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
    face_img = rgb_img[y:y+h, x:x+w]
    
    return face_img, masked_face, img, None

# Function to extract even-lit face areas (avoiding shadows and highlights)
def extract_even_lit_areas(face_img, masked_face):
    if face_img is None:
        return None
    
    # Convert to LAB color space
    lab_img = cv2.cvtColor(masked_face, cv2.COLOR_RGB2LAB)
    
    # Extract L channel (luminance)
    l_channel = lab_img[:,:,0]
    
    # Keep only pixels with moderate luminance (avoiding shadows and highlights)
    min_lum = 50  # Avoid deep shadows
    max_lum = 200  # Avoid highlights
    
    # Create mask for even-lit areas
    even_mask = np.logical_and(l_channel >= min_lum, l_channel <= max_lum)
    even_mask = np.logical_and(even_mask, masked_face[:,:,0] > 0)  # Combine with face mask
    
    # Apply mask to get even-lit face areas
    even_lit_face = np.zeros_like(masked_face)
    even_lit_face[even_mask] = masked_face[even_mask]
    
    return even_lit_face

# Function to determine skin undertone
def analyze_skin_undertone(even_lit_face):
    if even_lit_face is None:
        return "Unknown"
    
    # Extract only the non-black pixels
    pixels = even_lit_face[np.any(even_lit_face != [0, 0, 0], axis=-1)]
    
    if len(pixels) == 0:
        return "Unknown"
    
    # Calculate average RGB
    avg_color = np.mean(pixels, axis=0)
    r, g, b = avg_color
    
    # Analyze red vs blue channels to determine warm vs cool
    if r > b:
        return "Warm"
    else:
        return "Cool"

# Function to analyze skin contrast level
def analyze_contrast(even_lit_face):
    if even_lit_face is None:
        return "Unknown"
    
    # Extract only the non-black pixels
    pixels = even_lit_face[np.any(even_lit_face != [0, 0, 0], axis=-1)]
    
    if len(pixels) == 0:
        return "Unknown"
    
    # Convert to grayscale
    gray_pixels = np.dot(pixels, [0.299, 0.587, 0.114])
    
    # Calculate standard deviation of luminance
    std_dev = np.std(gray_pixels)
    
    # Classify contrast
    if std_dev < 15:
        return "Low"
    elif std_dev < 30:
        return "Medium"
    else:
        return "High"

# Function to extract dominant colors
def extract_skin_colors(even_lit_face, n_colors=5):
    if even_lit_face is None:
        return []
    
    # Extract only the non-black pixels
    mask = np.any(even_lit_face != [0, 0, 0], axis=-1)
    pixels = even_lit_face[mask].reshape(-1, 3)
    
    if len(pixels) == 0:
        return []
    
    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    
    # Convert to integer RGB values
    colors = colors.astype(int)
    
    return colors

# Function to get complementary colors
def get_complementary_colors(skin_colors, undertone, contrast):
    complementary_colors = []
    
    # Color palettes based on undertone and contrast
    warm_palette = {
        "Low": [
            (255, 166, 158),  # Peach
            (255, 214, 153),  # Light Gold
            (204, 153, 102),  # Camel
            (153, 204, 153),  # Sage Green
            (204, 204, 153)   # Olive
        ],
        "Medium": [
            (204, 102, 0),    # Burnt Orange
            (204, 153, 0),    # Gold
            (153, 102, 51),   # Brown
            (102, 153, 102),  # Forest Green
            (153, 102, 102)   # Terracotta
        ],
        "High": [
            (204, 51, 0),     # Tomato Red
            (153, 102, 0),    # Dark Gold
            (102, 51, 0),     # Deep Brown
            (51, 102, 51),    # Deep Forest Green
            (102, 51, 51)     # Burgundy
        ]
    }
    
    cool_palette = {
        "Low": [
            (204, 204, 255),  # Lavender
            (153, 204, 204),  # Powder Blue
            (153, 153, 204),  # Periwinkle
            (204, 153, 204),  # Lilac
            (204, 204, 204)   # Silver
        ],
        "Medium": [
            (102, 102, 204),  # Blue-Purple
            (102, 153, 153),  # Teal
            (102, 102, 153),  # Purple-Gray
            (153, 102, 153),  # Mauve
            (153, 153, 153)   # Gray
        ],
        "High": [
            (51, 51, 153),    # Royal Blue
            (0, 102, 102),    # Dark Teal
            (51, 51, 102),    # Navy
            (102, 51, 102),   # Purple
            (51, 51, 51)      # Charcoal
        ]
    }
    
    # Get appropriate palette
    if undertone == "Warm":
        palette = warm_palette.get(contrast, warm_palette["Medium"])
    else:
        palette = cool_palette.get(contrast, cool_palette["Medium"])
    
    for color in palette:
        complementary_colors.append(color)
    
    return complementary_colors

# Function to explain color recommendations
def explain_recommendations(complementary_colors, undertone, contrast):
    explanations = []
    
    color_descriptions = {
        # Warm colors
        (255, 166, 158): "Soft Peach - enhances natural warmth in skin with low contrast",
        (255, 214, 153): "Light Gold - brightens warm complexion without overwhelming",
        (204, 153, 102): "Camel - neutral that harmonizes with warm undertones",
        (153, 204, 153): "Sage Green - complements warm skin with a natural balance",
        (204, 204, 153): "Olive - earthy tone that enhances warm complexions",
        
        (204, 102, 0): "Burnt Orange - brings out the richness in medium contrast warm skin",
        (204, 153, 0): "Gold - highlights warmth in medium contrast complexions",
        (153, 102, 51): "Brown - grounds warm complexions with medium contrast",
        (102, 153, 102): "Forest Green - balances warmth with complementary depth",
        (153, 102, 102): "Terracotta - enhances natural flush in warm, medium contrast skin",
        
        (204, 51, 0): "Tomato Red - creates striking contrast that enhances high contrast warm skin",
        (153, 102, 0): "Dark Gold - adds richness to high contrast warm complexions",
        (102, 51, 0): "Deep Brown - anchors warm, high contrast coloring",
        (51, 102, 51): "Deep Forest Green - provides depth for high contrast warm skin",
        (102, 51, 51): "Burgundy - adds dramatic complement to warm, high contrast features",
        
        # Cool colors
        (204, 204, 255): "Lavender - softly enhances cool undertones with low contrast",
        (153, 204, 204): "Powder Blue - brightens cool, low contrast complexions",
        (153, 153, 204): "Periwinkle - adds gentle color that harmonizes with cool skin",
        (204, 153, 204): "Lilac - complements coolness while adding a touch of warmth",
        (204, 204, 204): "Silver - reflects light to enhance cool, low contrast features",
        
        (102, 102, 204): "Blue-Purple - brings depth to cool, medium contrast skin",
        (102, 153, 153): "Teal - balances cool undertones with medium contrast",
        (102, 102, 153): "Purple-Gray - sophisticated neutral for cool skin",
        (153, 102, 153): "Mauve - bridges cool and warm elements for medium contrast",
        (153, 153, 153): "Gray - classic neutral that enhances cool undertones",
        
        (51, 51, 153): "Royal Blue - creates striking complement to cool, high contrast features",
        (0, 102, 102): "Dark Teal - adds rich depth to cool, high contrast complexions",
        (51, 51, 102): "Navy - grounds cool coloring with sophisticated depth",
        (102, 51, 102): "Purple - dramatic color that enhances cool, high contrast skin",
        (51, 51, 51): "Charcoal - adds definition to cool, high contrast features"
    }
    
    # Get explanations for each color
    for i, color in enumerate(complementary_colors):
        # Convert tuple to rgb
        color_tuple = tuple(color)
        hex_color = rgb_to_hex(color)
        
        explanation = color_descriptions.get(color_tuple, f"Color {i+1} ({hex_color})")
        if color_tuple not in color_descriptions:
            # Add generic explanation based on undertone and contrast
            if undertone == "Warm":
                explanation += " - complements your warm undertones"
            else:
                explanation += " - enhances your cool undertones"
                
            if contrast == "Low":
                explanation += " with subtle harmony"
            elif contrast == "Medium":
                explanation += " with balanced contrast"
            else:
                explanation += " with striking definition"
        
        explanations.append(explanation)
    
    return explanations

# Root route - render a simple HTML form for testing
@app.route('/', methods=['GET'])
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Color Analysis API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                background-color: #f9f9f9;
            }
            .btn {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
            .results {
                margin-top: 20px;
                display: none;
            }
            .color-square {
                width: 50px;
                height: 50px;
                display: inline-block;
                margin-right: 10px;
                border: 1px solid #000;
            }
            #imagePreview {
                max-width: 300px;
                max-height: 300px;
            }
            .flex-row {
                display: flex;
                gap: 10px;
                align-items: center;
            }
        </style>
    </head>
    <body>
        <h1>Color Analysis API</h1>
        
        <div class="container">
            <div class="card">
                <h2>Upload Image</h2>
                <input type="file" id="imageInput" accept="image/*">
                <p>Select an image with a face to analyze skin tone and get color recommendations.</p>
                <img id="imagePreview" style="display: none;">
                <button class="btn" id="analyzeBtn">Analyze Image</button>
            </div>
            
            <div class="card results" id="resultsContainer">
                <h2>Analysis Results</h2>
                <div id="loadingIndicator">Analyzing image...</div>
                
                <div id="resultsContent" style="display: none;">
                    <p><strong>Undertone:</strong> <span id="undertone"></span></p>
                    <p><strong>Contrast:</strong> <span id="contrast"></span></p>
                    
                    <h3>Dominant Skin Colors:</h3>
                    <div id="skinColors"></div>
                    
                    <h3>Recommended Colors:</h3>
                    <div id="recommendedColors"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>API Documentation</h2>
                <p>Access the API documentation to learn more about available endpoints and how to use them.</p>
                <a href="/api/docs" class="btn">View API Docs</a>
            </div>
        </div>

        <script>
            document.getElementById('imageInput').addEventListener('change', function(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const img = document.getElementById('imagePreview');
                        img.src = e.target.result;
                        img.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            document.getElementById('analyzeBtn').addEventListener('click', function() {
                const imageInput = document.getElementById('imageInput');
                if (!imageInput.files || imageInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                const file = imageInput.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const resultsContainer = document.getElementById('resultsContainer');
                    const loadingIndicator = document.getElementById('loadingIndicator');
                    const resultsContent = document.getElementById('resultsContent');
                    
                    resultsContainer.style.display = 'block';
                    loadingIndicator.style.display = 'block';
                    resultsContent.style.display = 'none';
                    
                    // Send the image for analysis
                    fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: e.target.result
                        })
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('API request failed');
                        }
                        return response.json();
                    })
                    .then(data => {
                        loadingIndicator.style.display = 'none';
                        resultsContent.style.display = 'block';
                        
                        // Display results
                        document.getElementById('undertone').textContent = data.undertone;
                        document.getElementById('contrast').textContent = data.contrast;
                        
                        // Display skin colors
                        const skinColorsDiv = document.getElementById('skinColors');
                        skinColorsDiv.innerHTML = '';
                        data.skin_colors.forEach(color => {
                            const colorDiv = document.createElement('div');
                            colorDiv.className = 'flex-row';
                            
                            const colorSquare = document.createElement('div');
                            colorSquare.className = 'color-square';
                            colorSquare.style.backgroundColor = color.hex;
                            
                            const colorText = document.createElement('span');
                            colorText.textContent = color.hex;
                            
                            colorDiv.appendChild(colorSquare);
                            colorDiv.appendChild(colorText);
                            skinColorsDiv.appendChild(colorDiv);
                        });
                        
                        // Display recommended colors
                        const recommendedColorsDiv = document.getElementById('recommendedColors');
                        recommendedColorsDiv.innerHTML = '';
                        data.recommended_colors.forEach(color => {
                            const colorDiv = document.createElement('div');
                            colorDiv.className = 'flex-row';
                            
                            const colorSquare = document.createElement('div');
                            colorSquare.className = 'color-square';
                            colorSquare.style.backgroundColor = color.hex;
                            
                            const colorText = document.createElement('div');
                            colorText.textContent = `${color.hex} - ${color.explanation}`;
                            
                            colorDiv.appendChild(colorSquare);
                            colorDiv.appendChild(colorText);
                            recommendedColorsDiv.appendChild(colorDiv);
                        });
                    })
                    .catch(error => {
                        loadingIndicator.textContent = 'Error: ' + error.message;
                        console.error('Error:', error);
                    });
                };
                
                reader.readAsDataURL(file);
            });
        </script>
    </body>
    </html>
    '''
    return html

# Main API endpoint for color analysis
@app.route('/analyze', methods=['POST'])
def analyze_face_and_recommend_colors():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Get the base64 image from the request
        img_base64 = request.json['image']
        
        # Convert the base64 image to CV2 format
        img = base64_to_cv2(img_base64)
        
        # Detect face in the image
        face_img, masked_face, original_img, error = detect_face(img)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Extract even-lit face areas
        even_lit_face = extract_even_lit_areas(face_img, masked_face)
        
        if even_lit_face is None:
            return jsonify({'error': 'Could not extract evenly lit facial areas'}), 400
        
        # Analyze skin undertone
        undertone = analyze_skin_undertone(even_lit_face)
        
        # Analyze contrast level
        contrast = analyze_contrast(even_lit_face)
        
        # Extract skin colors
        skin_colors = extract_skin_colors(even_lit_face)
        skin_colors_hex = [rgb_to_hex(color) for color in skin_colors]
        
        # Get complementary colors
        complementary_colors = get_complementary_colors(skin_colors, undertone, contrast)
        complementary_colors_hex = [rgb_to_hex(color) for color in complementary_colors]
        
        # Get explanations for each color
        explanations = explain_recommendations(complementary_colors, undertone, contrast)
        
        # Prepare the response
        response = {
            'undertone': undertone,
            'contrast': contrast,
            'skin_colors': [
                {'rgb': color.tolist(), 'hex': hex_color} 
                for color, hex_color in zip(skin_colors, skin_colors_hex)
            ],
            'recommended_colors': [
                {'rgb': list(color), 'hex': hex_color, 'explanation': explanation} 
                for color, hex_color, explanation in zip(complementary_colors, complementary_colors_hex, explanations)
            ]
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Color analysis API is running'}), 200

if __name__ == '__main__':
    # Check if the shape predictor model exists
    if not os.path.exists(PREDICTOR_PATH):
        print(f"Warning: The shape predictor model file '{PREDICTOR_PATH}' does not exist.")
        print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 