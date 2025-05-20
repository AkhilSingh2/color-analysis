import requests
import base64
import os
import argparse
import json

def test_api(image_path, api_url="http://localhost:5000"):
    """
    Test the color analysis API with an image file.
    
    Args:
        image_path (str): Path to the image file
        api_url (str): Base URL of the API
    
    Returns:
        dict: API response
    """
    if not os.path.exists(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return None
    
    # Check health endpoint
    try:
        health_response = requests.get(f"{api_url}/health")
        print(f"Health check status: {health_response.status_code}")
        print(health_response.json())
    except requests.RequestException as e:
        print(f"Error checking API health: {e}")
        return None
    
    # Read and encode the image
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading or encoding image: {e}")
        return None
    
    # Send the request to the API
    try:
        print(f"Sending image to API for analysis...")
        response = requests.post(
            f"{api_url}/analyze",
            json={"image": encoded_string},
            headers={"Content-Type": "application/json"},
            timeout=30  # Increased timeout for face detection
        )
        
        if response.status_code != 200:
            print(f"API error (status code {response.status_code}): {response.text}")
            return None
        
        result = response.json()
        
        # Pretty print the results
        print("\n===== Color Analysis Results =====")
        print(f"Undertone: {result['undertone']}")
        print(f"Contrast: {result['contrast']}")
        
        print("\nDominant Skin Colors:")
        for i, color in enumerate(result['skin_colors'], 1):
            print(f"  Color {i}: RGB{color['rgb']} - HEX: {color['hex']}")
        
        print("\nRecommended Colors:")
        for i, color in enumerate(result['recommended_colors'], 1):
            print(f"  {i}. {color['hex']} - {color['explanation']}")
        
        return result
    
    except requests.RequestException as e:
        print(f"Error communicating with API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding API response: {response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the color analysis API")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument("--api-url", default="http://localhost:5000", help="Base URL of the API")
    
    args = parser.parse_args()
    test_api(args.image_path, args.api_url) 