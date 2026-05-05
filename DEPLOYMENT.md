# Deployment Guide

## GitHub Repository Setup

### Prerequisites
- GitHub account
- Git installed locally

### Steps

1. **Create Repository on GitHub**
   - Go to https://github.com/new
   - Repository name: `kastack-conversation-rag`
   - Description: "RAG System for Conversation Analysis with Topic Detection and User Persona Extraction"
   - Public
   - Do NOT initialize with README (already have one)
   - Click "Create repository"

2. **Connect Local Repository to GitHub**
   ```bash
   cd /path/to/KaStack
   git remote add origin https://github.com/YOUR_USERNAME/kastack-conversation-rag.git
   git branch -M main
   git push -u origin main
   ```

3. **Verify Push**
   ```bash
   git remote -v
   # Should show:
   # origin  https://github.com/YOUR_USERNAME/kastack-conversation-rag.git (fetch)
   # origin  https://github.com/YOUR_USERNAME/kastack-conversation-rag.git (push)
   ```

## Heroku Deployment

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed

### Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create kastack-conversation-rag
   ```
   (Choose a unique name if this is taken)

3. **Set Environment Variables** (if needed)
   ```bash
   heroku config:set FLASK_ENV=production
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **View Logs**
   ```bash
   heroku logs --tail
   ```

6. **Open App**
   ```bash
   heroku open
   ```

### Troubleshooting

**Deployment fails with "no module named"**:
- Check `requirements.txt` has all dependencies
- Run locally: `pip install -r requirements.txt`

**App crashes on startup**:
- Check logs: `heroku logs --tail`
- Verify Flask app starts locally: `python backend/app.py`

**API endpoints return 404**:
- Ensure backend/app.py is properly configured
- Check CORS settings for frontend requests

## Vercel Deployment (Alternative)

### Steps

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy Frontend Only**
   ```bash
   cd backend
   vercel --prod
   ```

3. **Update Backend URL in index.html**
   - Change `API_URL` to your deployed backend URL

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "backend/app.py"]
```

### Build and Run

```bash
docker build -t kastack-rag .
docker run -p 5000:5000 kastack-rag
```

## Testing Deployment

After deployment, test endpoints:

```bash
# Health check
curl https://YOUR_APP.herokuapp.com/health

# Get stats
curl https://YOUR_APP.herokuapp.com/api/stats

# Chat
curl -X POST https://YOUR_APP.herokuapp.com/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of person is this user?"}'
```

## Performance Optimization

### For Production:
1. Use gunicorn with 4 workers: `gunicorn -w 4 app:app`
2. Enable gzip compression
3. Cache RAG indexes in memory
4. Use CDN for static files
5. Monitor with New Relic or similar

### Scaling:
- Use Heroku Dyos with more resources for larger datasets
- Consider AWS Lambda + API Gateway for serverless
- Use Redis for caching retrieved contexts
