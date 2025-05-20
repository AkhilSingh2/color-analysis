# Color Analysis API

This API analyzes facial images to determine skin undertone, contrast level, and recommend complementary colors.

## Features

- Face detection and extraction
- Skin undertone analysis (Warm/Cool)
- Contrast level determination (Low/Medium/High)
- Skin color extraction
- Personalized color recommendations with explanations
- Interactive API documentation (Swagger UI)
- Web interface for testing

## Setup and Installation

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

### Installation (Local)

1. Clone this repository:
```
git clone https://github.com/yourusername/color-analysis-api.git
cd color-analysis-api
```

2. Install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the shape predictor model:
```
curl -L -O https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

4. Run the application:
```
python app.py
```

The API will be available at `http://localhost:5000`.

### Installation (Docker)

1. Build and run the Docker container:
```
docker-compose up -d
```

The API will be available at `http://localhost:5001`.

## API Usage

### Web Interface

Access the web interface at `http://localhost:5001/` (or your deployment URL) to:
- Upload and analyze images
- View analysis results
- Access API documentation

### API Documentation

API documentation is available at `http://localhost:5001/api/docs/` (or your deployment URL).

### Analyze Face and Recommend Colors

**Endpoint:** `/analyze`

**Method:** POST

**Request Body:**
```json
{
  "image": "base64_encoded_image_data"
}
```

The image should be encoded as a base64 string. If it includes the data URL prefix (`data:image/jpeg;base64,`), it will be automatically processed.

**Response:**
```json
{
  "undertone": "Warm",
  "contrast": "Medium",
  "skin_colors": [
    {
      "rgb": [233, 185, 157],
      "hex": "#e9b99d"
    },
    ...
  ],
  "recommended_colors": [
    {
      "rgb": [204, 102, 0],
      "hex": "#cc6600",
      "explanation": "Burnt Orange - brings out the richness in medium contrast warm skin"
    },
    ...
  ]
}
```

### Health Check

**Endpoint:** `/health`

**Method:** GET

**Response:**
```json
{
  "status": "healthy",
  "message": "Color analysis API is running"
}
```

## Deployment

### Railway

For deploying to Railway, see [RAILWAY.md](RAILWAY.md) for detailed instructions.

### Other Deployment Options

- **Render**: Easy Docker deployment with free tier
- **Fly.io**: Distributed deployment with free tier
- **Google Cloud Run**: Serverless containers with free tier
- **Heroku**: Platform-as-a-service with container support

## Example Client Code (JavaScript)

```javascript
async function analyzeImage(imageFile) {
  // Convert the image file to base64
  const base64Image = await fileToBase64(imageFile);
  
  // Send the request to the API
  const response = await fetch('http://localhost:5001/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64Image
    })
  });
  
  return await response.json();
}

// Helper function to convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
}
```

## Example Client Code (Python)

```python
import requests
import base64

def analyze_image(image_path):
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Send the request to the API
    response = requests.post(
        "http://localhost:5001/analyze",
        json={"image": encoded_string}
    )
    
    return response.json()

# Example usage
result = analyze_image("path/to/your/image.jpg")
print(result)
```

## Testing the API

Use the included test script:

```bash
python test_api.py path/to/your/image.jpg
```

## License

[MIT License](LICENSE) 