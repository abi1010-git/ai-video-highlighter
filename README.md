# AI Video Highlighter

Upload a video, transcribe it, rank the most relevant timestamped moments, and create a jumpcut edit that skips interruptions, breaks, and low-importance sections.

## Streamlit Website

Run the Streamlit version locally:

```powershell
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

The Streamlit app shows:

- the full lecture video
- a second jumpcut edit below it
- timestamped highlight cards that jump the full lecture to important moments
- a download button for the jumpcut MP4

### Deploy on Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to https://share.streamlit.io.
3. Create a new app from your repo.
4. Set the main file path to:

   ```text
   streamlit_app.py
   ```

5. In the app settings, add this secret:

   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

`packages.txt` installs FFmpeg for the hosted app.

### Deploy on Render

This repo includes `render.yaml`, so Render can create the service from your GitHub repo.

1. Push this folder to GitHub.
2. In Render, choose **New Blueprint**.
3. Select this repository.
4. Add `OPENAI_API_KEY` as an environment variable.

Render will use `Dockerfile.streamlit` and publish the Streamlit website.

## Run in VS Code

1. Open this folder in VS Code:

   ```powershell
   code C:\Users\abhia\ai-video-highlighter\ai-video-highlighter
   ```

2. Open the VS Code terminal and activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`, then paste your OpenAI API key:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Start the web app:

   ```powershell
   python app.py
   ```

6. Open http://127.0.0.1:5000 in your browser.

## Run with Docker

1. Create your `.env` file if you want automatic transcription:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Build and start the website:

   ```powershell
   docker compose up --build
   ```

3. Open http://127.0.0.1:5050 in your browser.

Docker keeps your uploaded videos, cached transcripts, and generated outputs in these local folders:

- `uploads/`
- `transcripts/`
- `outputs/`

Stop the website with:

```powershell
docker compose down
```

## Use without an API key

Upload a timestamped `.srt`, `.vtt`, or `.json` transcript with the video. The app can rank those transcript moments without calling the transcription API.

## Train A Small Highlight Model

This project can train a small local model from lecture transcripts. Use only transcripts you own or are allowed to reuse. Good starter sources are open course sites such as Open Yale Courses or MIT OpenCourseWare, but always check each course page and license terms.

1. Copy the example URL list:

   ```powershell
   Copy-Item data\transcript_urls.example.txt data\transcript_urls.txt
   ```

2. Edit `data\transcript_urls.txt` and keep only transcript pages you want to use.

3. Download the transcript text:

   ```powershell
   python download_transcripts.py
   ```

4. Train the small classifier:

   ```powershell
   python train_highlight_model.py
   ```

5. Run the Streamlit app:

   ```powershell
   streamlit run streamlit_app.py
   ```

If `models/highlight_model.joblib` exists, the Streamlit sidebar will let you use the trained model. The model is a lightweight TF-IDF + logistic regression classifier trained with weak labels, so it is a starter model. It improves when you add better transcripts or replace the generated labels in `data/training_examples.csv` with your own 1/0 labels.

## Command line

```powershell
python main.py lecture.mp4 --query "exam tips"
```

With an existing transcript:

```powershell
python main.py lecture.mp4 --transcript lecture.srt --query "gradient descent"
```

With a trained model:

```powershell
python main.py lecture.mp4 --transcript lecture.srt --model models/highlight_model.joblib
```

To export a short highlight reel:

```powershell
python main.py lecture.mp4 --query "important definitions" --make-reel
```

The exported reel is a jumpcut edit: it keeps the selected important moments, adds a few seconds of context around each one, merges nearby clips, and skips the rest.

## Files

- `app.py` runs the upload/search web app.
- `streamlit_app.py` runs the Streamlit website.
- `main.py` runs the transcript highlighter from the terminal.
- `highlighter.py` contains the transcription, chunking, scoring, and clipping logic.
- `download_transcripts.py` downloads transcript pages listed in `data/transcript_urls.txt`.
- `train_highlight_model.py` trains the small highlight model.
- `highlight_model.py` contains the training and prediction code.
- `visual_yolo_highlighter.py` keeps the original object-detection approach for sports or action clips.
