# Deploying LeafLens to Render

This guide will walk you through deploying your LeafLens Flask application to Render.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Render account (sign up at [render.com](https://render.com))

## Step 1: Prepare Your Repository

### 1.1 Ensure Required Files Are Present

Make sure these files exist in your `App/` directory:
- ✅ `Procfile` - Tells Render how to run your app
- ✅ `requirements.txt` - Lists all Python dependencies
- ✅ `runtime.txt` - Specifies Python version (optional but recommended)
- ✅ `app.py` - Your Flask application
- ✅ `trained_model.pth` - Your trained model file (IMPORTANT: This is large ~100MB+)

### 1.2 Important Notes About File Sizes

⚠️ **PyTorch and Model Files are Large:**
- PyTorch installation: ~2-3 GB
- `trained_model.pth`: ~100-500 MB (depending on your model)
- Total build size may exceed 5 GB

**Render Free Tier Limitations:**
- Free tier has build timeout limits
- Large builds may take 30-60 minutes
- Consider using Render's paid plans for faster builds

### 1.3 Verify Your Project Structure

Your repository should look like this:
```
CultiKure-Disease-Prediction/
├── App/
│   ├── app.py
│   ├── Procfile
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── trained_model.pth
│   ├── disease_info.csv
│   ├── supplement_info.csv
│   ├── static/
│   │   └── uploads/
│   └── templates/
└── training_results/  (optional, for documentation)
```

## Step 2: Push to GitHub

1. Initialize git repository (if not already done):
```bash
cd CultiKure-Disease-Prediction
git init
git add .
git commit -m "Initial commit - Ready for Render deployment"
```

2. Create a new repository on GitHub

3. Push your code:
```bash
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Render

### 3.1 Create a New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if not already connected
4. Select your repository: `CultiKure-Disease-Prediction`

### 3.2 Configure Build Settings

**Name:** `leaflens` (or any name you prefer)

**Region:** Choose closest to your users (e.g., `Oregon (US West)`)

**Branch:** `main` (or your default branch)

**Root Directory:** ⚠️ **IMPORTANT** - Set this to `App`
   - Render will look for files in the `App/` directory
   - This is crucial because your `Procfile` and `app.py` are in `App/`

**Runtime:** `Python 3`

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn app:app
```
(Or leave blank - Render will use the Procfile)

### 3.3 Environment Variables (Optional)

You can add environment variables if needed:
- `FLASK_ENV=production` (to disable debug mode)
- `PORT=10000` (Render sets this automatically, but you can override)

### 3.4 Plan Selection

- **Free Tier:** 
  - ✅ Free forever
  - ⚠️ Spins down after 15 minutes of inactivity
  - ⚠️ Slower builds (30-60 min for PyTorch)
  - ⚠️ Limited resources

- **Starter Plan ($7/month):**
  - ✅ Always on
  - ✅ Faster builds
  - ✅ Better performance
  - ✅ Recommended for production

### 3.5 Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. Monitor the build logs - this will take 30-60 minutes due to PyTorch installation

## Step 4: Monitor Deployment

### 4.1 Build Logs

Watch the build logs for:
- ✅ Python installation
- ✅ Dependencies installation (this is the slow part)
- ✅ Model file upload
- ✅ Application startup

### 4.2 Common Issues

**Issue: Build Timeout**
- **Solution:** Upgrade to paid plan or optimize requirements.txt

**Issue: "Module not found"**
- **Solution:** Check that all dependencies are in `requirements.txt`

**Issue: "Model file not found"**
- **Solution:** Ensure `trained_model.pth` is committed to git and in the `App/` directory

**Issue: "Port already in use"**
- **Solution:** Render handles this automatically, but ensure your Procfile is correct

**Issue: "Static files not loading"**
- **Solution:** Check that static files are in `App/static/` directory

## Step 5: Post-Deployment

### 5.1 Verify Your App

Once deployed, your app will be available at:
```
https://your-app-name.onrender.com
```

Test these endpoints:
- ✅ Home page: `/`
- ✅ AI Engine: `/index`
- ✅ Documentation: `/documentation`
- ✅ Market: `/market`
- ✅ Contact: `/contact`

### 5.2 Custom Domain (Optional)

1. Go to your service settings
2. Click **"Custom Domains"**
3. Add your domain
4. Follow DNS configuration instructions

### 5.3 Auto-Deploy

Render automatically deploys when you push to your main branch. You can:
- Disable auto-deploy in settings
- Set up manual deploys only
- Configure branch-specific deployments

## Step 6: Optimize for Production

### 6.1 Update Requirements (Optional)

Consider pinning exact versions for stability:
```txt
Flask==3.1.2
torch==2.9.1
torchvision==0.24.1
# ... etc
```

### 6.2 Add .gitignore

Create `.gitignore` in your root:
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.env
*.pth
.DS_Store
```

**Note:** Don't ignore `trained_model.pth` - it's needed for deployment!

### 6.3 Performance Tips

1. **Enable Caching:** Render automatically caches static files
2. **Optimize Images:** Compress images in `static/uploads/`
3. **Database:** Consider adding a database for user uploads (Render PostgreSQL)
4. **CDN:** Use Render's CDN for static assets

## Troubleshooting

### Build Fails

1. Check build logs for specific errors
2. Verify all files are in correct directories
3. Ensure `Procfile` is in `App/` directory
4. Check Python version compatibility

### App Crashes

1. Check logs in Render dashboard
2. Verify model file is loaded correctly
3. Check file paths (they should be relative to `App/`)
4. Ensure all CSV files are present

### Slow Performance

1. Upgrade to paid plan for better resources
2. Optimize model loading (lazy loading)
3. Add caching for predictions
4. Use Render's PostgreSQL for data storage

## Support

- **Render Docs:** https://render.com/docs
- **Render Support:** support@render.com
- **Community:** https://community.render.com

## Quick Reference

**Procfile:**
```
web: gunicorn app:app
```

**Root Directory:** `App`

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:** (Auto-detected from Procfile)
```bash
gunicorn app:app
```

---

**Good luck with your deployment! 🚀**

