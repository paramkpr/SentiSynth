# SentiSynth: Synthetic Data Generation for Sentiment Analysis

SentiSynth explores how synthetic data can improve sentiment analysis models when labeled data is scarce.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentisynth.git
cd sentisynth

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt
```

## Usage

More details coming soon!

## Project Structure

- `sentisynth/`: Main package
  - `data/`: Data loading and processing
  - `models/`: Model implementations
  - `generation/`: Synthetic data generation
  - `evaluation/`: Evaluation metrics and analysis
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: Utility scripts 


## Training
To run on `weftdrive`: 
```bash
  nohup /srv/gpurun.pl python src/senti_synth/cli/01_train_teacher.py configs/teacher/stt2_hf.yaml > ~/scratch/senti_synth/logs/$(date +%Y%m%d_%H%M).log 2>&1 &
```

### Setting up on weftdrive
1. SSH into weftdrive: `ssh paramkapur@weftdrive.private.reed.edu`
2. Git clone the repository: `git clone https://github.com/paramkpr/senti_synth.git`
3. Setup the conda environment `/srv/conda/bin/conda init` and `source ~/.bashrc`
4. Enter the conda environment `conda activate deep-learning`
   1. Check what packages are installed `conda list`
   2. Install the packages for the project `pip install -r requirements.txt`
   3. Install the project `pip install -e .`
5. SCP `data/clean` to `weftdrive:~/scratch/data/clean`: `scp -r data/clean weftdrive:~/scratch/data/`
   1. Ensure that the config file points to the correct path: `dataset_path: "~/scratch/data/clean"`
6. Setup W&B: 
   1. `export WANDB_API_KEY="..."`
   2. `python -m wandb login`
7. Run the training script: `nohup /srv/gpurun.pl python src/senti_synth/cli/01_train_teacher.py configs/teacher/stt2_hf.yaml > ~/scratch/senti_synth/logs/$(date +%Y%m%d_%H%M).log 2>&1 &`

