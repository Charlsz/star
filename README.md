# STAR

Space Taxonomy and Analysis Recognition.

Upload a galaxy image and the model predicts whether it is smooth or featured/disk, then shows the confidence and class probabilities.

## Data source

This project uses data from the Galaxy Zoo 2 public releases.

Create the following folders inside `data/`:

- `data/raw/`
- `data/raw/galaxy_zoo_2/`

The raw files should be placed in `data/raw/galaxy_zoo_2/` like this:

```bash
data/
├── raw/
│   └── galaxy_zoo_2/
│       ├── gz2_hart16.csv
│       ├── gz2_filename_mapping.csv
│       └── images/
│           ├── image_1.jpg
│           ├── image_2.jpg
│           └── ...
└── processed/
```

Download the raw files from the following sources:

- **Morphology table:** `gz2_hart16.csv.gz` from the Galaxy Zoo data site  
  https://data.galaxyzoo.org/#section-8

- **Image files:** `images_gz2.zip` from the Galaxy Zoo 2 Zenodo release  
  https://zenodo.org/records/3565489

- **Image mapping file:** `gz2_filename_mapping.csv`, used to match image filenames with galaxy entries in the morphology table  
  https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1

After downloading:
- Place `gz2_hart16.csv.` in `data/raw/galaxy_zoo_2/`
- Place `gz2_filename_mapping.csv` in `data/raw/galaxy_zoo_2/`
- Extract `images_gz2.zip`
- Move the extracted `images/` folder into `data/raw/galaxy_zoo_2/`

The processed files used during training are saved in `data/processed/`.

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
   cd star
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

## How to train

Before running the app, you need to train the model to generate the saved weights file used for prediction.

Run:

```bash
python star/train.py
```

After training, the model weights will be saved as:

```bash
star_cnn.pt
```

## How to run

After the model has been trained, start the Streamlit app with:

```bash
python -m streamlit run star/app.py
```

Then open the local URL shown in the terminal, upload a galaxy image, and click **Predict**.