# Quick Deploy to Render - Summary

## Essential Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. **IMPORTANT Settings:**
   - **Root Directory:** `App` ⚠️ (This is critical!)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (or leave blank - uses Procfile)
   - **Plan:** Free tier (or Starter for better performance)

5. Click **"Create Web Service"**

### 3. Wait for Build
- ⏱️ Build time: 30-60 minutes (PyTorch is large)
- 📊 Monitor build logs
- ✅ Your app will be live at `https://your-app-name.onrender.com`

## Key Files Created

✅ `Procfile` - Tells Render how to run your app
✅ `runtime.txt` - Python version
✅ `render.yaml` - Alternative configuration (optional)
✅ `.gitignore` - Excludes unnecessary files

## Important Notes

⚠️ **Root Directory MUST be `App`** - Your app files are in the App/ folder

⚠️ **Large Build Size** - PyTorch + model = ~3-5 GB total

⚠️ **Free Tier Limitations:**
- Spins down after 15 min inactivity
- Slower builds
- Consider Starter plan ($7/month) for production

## Troubleshooting

**Build fails?** Check that:
- Root directory is set to `App`
- `Procfile` exists in `App/` directory
- `trained_model.pth` is committed to git
- All dependencies in `requirements.txt`

**App crashes?** Check logs in Render dashboard

## Full Guide

See `RENDER_DEPLOYMENT.md` for detailed instructions.

