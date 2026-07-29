# NSF Future Manufacturing Data Challenge Submission (Team RAV)

This repository holds the code, notebooks, and supporting material for our submission to the NSF Future Manufacturing Data Challenge. The task asks teams to predict spatially varying laser track geometry along the scan direction, with calibrated uncertainty, using thermal video, SEM imagery, and profilometer height maps from four directed energy deposition (DED) tracks, Track 8, Track 10, Track 14, and Track 21.

## Approach summary

Our pipeline predicts local track width, left and right boundary position, and a set of cross track contour descriptors (centerline position, peak depth, profile skewness, and profile waviness) as continuous functions of position along the scan direction, each with a calibrated uncertainty estimate.

All three model families (width, boundary position, and contour shape) use Gaussian process regression with an isotropic Matern 3/2 kernel. Per sample measurement noise, taken from the profilometer width standard deviation, is passed in directly as a known alpha term rather than fit as a free white noise kernel. That change was what fixed a severe overconfidence problem we ran into early on. We also tried an automatic relevance determination (ARD) kernel and a CNN as alternatives. The ARD kernel became unstable given our sample size of roughly 300 points per track and 13 free hyperparameters, and the CNN produced very poor negative log likelihood scores, so the isotropic kernel is what we kept.

The model inputs are a 14 feature, physically motivated set that includes cooling rate and linear energy density. Both were kept after a full ablation study since they improved MAE by roughly 30 percent. A thermal gradient proxy feature was dropped because it hurt MAE in every ablation combination we tried, and because Track 21 values for that feature fell almost entirely outside the training range.

## Key results

- Direct local width prediction reaches an MAE around 0.120 mm.
- Width recovered from independently predicted left and right boundaries has a noticeably higher MAE, around 0.228 mm, since those boundary models have to learn each track's positional offset from features that don't encode it directly.
- Isotropic Matern 3/2 GPR with known heteroscedastic noise clearly outperformed both the ARD kernel and the CNN baseline for this dataset size.

Full metrics, calibration plots, and per track breakdowns are in the report and in `ML_results/`.

## What is in this repository

The layout looks like this.

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── 01_starter_code_loading_and_visualization.ipynb
├── 02_starter_code_loading_and_visualization_standalone_colab.ipynb
├── Train_models.ipynb
├── Ablation studies.ipynb
├── src/
│   └── nsf_fmrg_data.py
├── scripts/
│   ├── feature_extraction.py
│   ├── gpr_width_model.py
│   ├── gpr_boundary_model.py
│   ├── gpr_contour_model.py
│   ├── cv_width_model.py
│   └── run_feature_ablation.py
├── ML_results/
├── Track_8/            (generated, not committed)
├── Track_10/           (generated, not committed)
├── Track_14/           (generated, not committed)
├── Track_21/           (generated, not committed)
└── data/               (not committed, downloaded separately)
```

A couple of notes on this tree. The exact file names under `scripts/` should match whatever is currently in your local `scripts` folder, since that folder was reorganized recently, so treat the list above as a best guess based on our earlier work and adjust it if anything doesn't line up. Also see the note below about the `Track_8`, `Track_10`, `Track_14`, and `Track_21` folders and why they aren't committed.

### `src/` versus `scripts/`

`src/nsf_fmrg_data.py` is the data loading helper from the organizer's starter code (thermal frame extraction, SEM tile loading, Wyko ASC height map parsing). It stays close to the original starter repository so anyone comparing our work against the baseline can see what changed.

`scripts/` holds our own pipeline code, feature extraction, the three GPR model families, cross validation, and the ablation study runner. Both folders are added to `sys.path` at the top of each notebook, using a pattern like this.

```python
scripts_path = Path("./scripts").resolve()
src_path = Path("./src").resolve()

for folder_path in [scripts_path, src_path]:
    import sys
    if str(folder_path) not in sys.path:
        sys.path.append(str(folder_path))
```

Because this resolves the paths relative to the current working directory, notebooks need to be launched from the repository root for the imports to work. If you open a notebook from a different working directory and the imports fail, that's almost always why.

## Data access

The raw multimodal dataset is hosted on Zenodo, not included in this repository because of its size.

Dataset DOI: https://doi.org/10.5281/zenodo.21285367

After downloading, extract the files into the repository using this layout.

```
data/raw/thermal/
  Thermal_8.mat
  Thermal_10.mat
  Thermal_14.mat
  Thermal_21.mat

data/raw/sem/
  SEM_8/PlainImages/
  SEM_10/PlainImages/
  SEM_14/PlainImages/
  SEM_21/PlainImages/

data/raw/height_maps/
  Heightmap_8.ASC
  Heightmap_10.ASC
  Heightmap_14.ASC
  Heightmap_21.ASC
```

This matches the layout used by the organizer's starter repository at https://github.com/abhishekhanchate/nsf-fmrg-data-challenge, so anything documented there about the raw files applies here too.

### About the `Track_8` / `Track_10` / `Track_14` / `Track_21` folders

These folders hold the processed thermal and height data produced by running `02_starter_code_loading_and_visualization_standalone_colab.ipynb` on the raw dataset above. That output feeds into feature extraction and model training.

We are not committing these folders to the repository and are adding them to `.gitignore` instead. Some of the individual processed files run as large as 250 MB, well past GitHub's 100 MB per file limit, so pushing them directly isn't an option without setting up Git LFS. Regenerating them locally avoids that entirely. Run the notebook once after downloading the raw data and it will regenerate all four folders.

## Environment setup

This project was built and tested with Python 3.10 or newer.

```bash
git clone https://github.com/Realone1110/NSF-Future-Manufacturing-Challenge
cd <repository folder>
python -m venv .venv
source .venv/bin/activate      # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` in this repository lists the packages we know the pipeline depends on, with loose minimum versions. For exact reproducibility of the environment we actually ran, it's worth also running `pip freeze > requirements.txt` from your working environment and comparing the two, since that captures the precise versions installed rather than a best guess.

## Reproducing the results

1. Download the raw dataset from Zenodo and place it under `data/raw` using the layout shown above.
2. Run `02_starter_code_loading_and_visualization_standalone_colab.ipynb` to generate the processed `Track_8`, `Track_10`, `Track_14`, and `Track_21` folders. This step produces both the thermal and height map data used downstream, saved into the same four folders.
3. Run `Train_models.ipynb` to extract features and train the three GPR model families (width, boundary, contour descriptors). This writes trained models, prediction outputs, and figures to `ML_results/`.
4. Run `Ablation studies.ipynb` to reproduce the feature ablation study and the kernel comparison experiments referenced in the report.
5. `01_starter_code_loading_and_visualization.ipynb` is the organizer's original participant guide notebook, kept for reference and not required for reproducing our results.

If you change any code in `scripts/` or `src/` while a notebook's kernel is running, restart the kernel fully rather than just re running cells, otherwise Python's import cache can silently keep the old version of the module in memory.

## Known limitations

- Boundary derived width has a noticeably higher MAE than direct width prediction, since the independent left and right boundary models have to learn each track's positional offset from features that don't encode it.
- Two contour descriptors, profile skewness and profile waviness, show high negative log likelihood due to a noise collapse effect in the GPR fit.
- Linear energy density is constant within a given track, so it is always out of distribution for the held out track under leave one track out cross validation.
- The mapping from laser power to track ID (200, 300, 350, and 400 watts corresponding to Track 8, 10, 14, and 21) is inferred from the ordering of parallel documentation rather than a confirmed lookup table. We're disclosing this assumption explicitly rather than presenting it as verified.

## Use of generative AI

Parts of this project, including code drafting, debugging, diagnostic experiments, the LaTeX report, and the presentation slides, were developed with help from an AI coding assistant (Claude, by Anthropic), under our direction and review. All modeling decisions, including kernel choice, feature selection, and the interpretation of results, were made and validated by the team.

## Citation

If referencing the dataset or the organizer's starter material outside of this challenge, cite the dataset paper and the Zenodo record.

```bibtex
@dataset{hanchate2026nsffmrgdedchallengedata,
  title        = {NSF Future Manufacturing Data Challenge: A Multimodal DED Dataset for Probabilistic Local Geometry Prediction in Laser Tracks},
  author       = {Hanchate, Abhishek and Balhara, Himanshu and Bukkapatnam, Satish T. S.},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21285367},
  url          = {https://doi.org/10.5281/zenodo.21285367}
}
```

The companion dataset paper is on arXiv at https://arxiv.org/abs/2607.07965.

## Data license and usage

The raw dataset is provided by the challenge organizers under terms described in their repository at https://github.com/abhishekhanchate/nsf-fmrg-data-challenge (see `DATA_USE_LICENSE.md` there). Use of the data here follows those terms. Code in this repository is our own work for the challenge submission; add a license of your choosing if you plan to make this repository public beyond the competition.
