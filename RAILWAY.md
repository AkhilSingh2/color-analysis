# Deploying to Railway

This document provides instructions for deploying the Color Analysis API to Railway.

## Prerequisites

- GitHub account
- Railway account (sign up at [railway.app](https://railway.app))

## Deployment Steps

### Option 1: Using Railway Web Interface (Recommended)

1. Log in to Railway using your GitHub account at [railway.app](https://railway.app)

2. Create a new project and select "Deploy from GitHub repo"

3. Connect your GitHub account and select this repository

4. Railway will automatically detect the Dockerfile and set up the deployment

5. In the Railway dashboard, go to the "Variables" tab and add the following environment variables:
   - `PORT`: `5000`
   - `PREDICTOR_PATH`: `/app/models/shape_predictor_68_face_landmarks.dat`

6. Railway will automatically deploy your application. Once the deployment is complete, you can access your API using the provided URL

7. To add a custom domain, go to the "Settings" tab and click on "Add Domain"

### Option 2: Using Railway CLI

If you prefer using the command line:

1. Install the Railway CLI:
   ```
   npm install -g @railway/cli
   ```

2. Log in to Railway:
   ```
   railway login
   ```

3. Link your project:
   ```
   railway link
   ```

4. Deploy the project:
   ```
   railway up
   ```

5. Open the project dashboard:
   ```
   railway open
   ```

## Monitoring and Logging

- View logs by going to the "Deployments" tab and clicking on the latest deployment
- Set up monitors in the "Monitors" tab to track your application's health

## Scaling

The free tier includes:
- 5 USD credit per month
- 512 MB RAM
- Shared CPU

If you need more resources, you can upgrade to a paid plan.

## Troubleshooting

1. If the deployment fails, check the logs for any errors

2. Common issues:
   - Memory limits: The color analysis process can be memory-intensive. If you encounter memory errors, you may need to optimize the code or upgrade to a plan with more memory
   - Missing environment variables: Ensure all required environment variables are set
   - Timeouts: Railway has a default timeout for requests. Long-running image processing might need optimization

3. Visit [docs.railway.app](https://docs.railway.app) for more information 