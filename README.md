# STAR

A galaxy morphology classification app built with PyTorch and Streamlit.  
Upload a galaxy image and the model predicts whether it is smooth or featured/disk, then shows the confidence and class probabilities.

## Features

- Upload a galaxy image through a simple web interface.
- Predict the galaxy morphology class.
- Display confidence for the predicted class.
- Show probabilities for all available classes.
- Run locally with Streamlit.

## Data source

This project uses data from the Galaxy Zoo 2 public releases.

- **Morphology table:** `gz2_hart16.csv.gz` from the Galaxy Zoo data site  
  https://data.galaxyzoo.org/#section-8

- **Image files:** `images_gz2.zip` from the Galaxy Zoo 2 Zenodo release  
  https://zenodo.org/records/3565489

- **Image mapping file:** `gz2_filename_mapping.csv`, used to match image filenames with galaxy entries in the morphology table  
  https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1

The Galaxy Zoo 2 images are based on galaxies from the Sloan Digital Sky Survey (SDSS).

## How it works

The app loads a trained image classification model and processes an uploaded galaxy image before making a prediction.  
After inference, it returns the predicted class, the confidence score, and the probability for each class.

Current classes:
- Smooth
- Featured/Disk

## How to install

1. Clone the repository:
   ```bash
   git clone https://github.com/Charlsz/star.git
   cd https://github.com/Charlsz/star.git
   ```

2. Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to run

Start the Streamlit app with:

```bash
python -m streamlit run star/app.py
```

Then open the local URL shown in the terminal, upload a galaxy image, and click **Predict**.