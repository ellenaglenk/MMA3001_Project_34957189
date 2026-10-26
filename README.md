# MMA3001 - Average 2D Tree Structure Modelling

## 1. Project Overview

This project investigates the use of numerical methods and
machine learning to analyse tree images and produce a
representative average two-dimensional (2D) tree structure.

The intended application is to understand typical tree geometry
that could inform the design of agricultural machinery.

The project uses RGB images, encoded depth images, and
ground-truth tree structure annotations collected from orchard
trees across summer and winter.

The final outcome is a representative 2D tree model of the apple trees in this particular field.

## 2. Project Objectives

The objectives are to:

1. Organise and preprocess a dataset of orchard tree images.
2. Investigate numerical interpolation of missing depth data.
3. Extract geometric information from annotated tree structures.
4. Develop and compare machine learning methods.
5. Train a U-Net neural network to predict tree structure masks.
6. Evaluate predictions using held-out validation trees.
7. Investigate whether the model transfers between seasons.
8. Combine predicted structures to construct an average 2D tree.

## 3. Dataset

The dataset contains observations of orchard trees captured
during summer and winter.

Image types include:

- RGB: Colour photographs of trees.
- D: Encoded depth images.
- GT: Ground-truth binary annotations of tree structures.

Image filenames identify the orchard row, tree number, season,
image type, and observation identifier.

The dataset contains:

- 562 unique trees.
- 562 summer RGB-depth observations.
- 562 winter RGB-depth observations.
- 562 summer ground-truth structure masks.

The project separates physical trees into training and
held-out validation groups to prevent the same tree appearing
in both groups.

The original dataset must be provided separately. Do not
assume it is included in this GitHub repository.

## 4. Software Environment

The project was developed in Google Colab using Python 3.

Recommended environment:

- Google Colab or a compatible Jupyter notebook.
- Python 3.
- Google Drive access for the dataset.
- Internet access to install dependencies if required.
- Optional GPU acceleration for neural network training.

### Python Dependencies

The project uses libraries including:

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- Pillow
- tensorflow
- opencv-python

Additional packages may be required by individual exploratory
notebook sections.

Install the main dependencies in Colab with:

```python
%pip install numpy pandas matplotlib scipy scikit-learn Pillow tensorflow opencv-python
```

Google Colab may already provide many of these packages.

## 5. Repository Contents

The main project notebook is:

`MMA3001_Assignment.ipynb`

The notebook contains the dataset investigation,
preprocessing, numerical methods, model development,
evaluation, and final average-tree generation.

Generated results may include:

- Processed dataset metadata.
- Extracted tree geometry features.
- Model comparison results.
- Validation metrics and plots.
- Trained neural network files.
- Predicted segmentation masks.
- Average 2D tree visualisations.

Generated files are generally written to
`/content/processed_data/` during Colab execution.

Files stored only in `/content/` are temporary and may be
lost when the Colab runtime is reset.

## 6. Instructions for Running the Project

### Step 1 - Obtain the Dataset

Place the supplied tree dataset ZIP file in Google Drive.

The original project used:

`/content/drive/MyDrive/2026/Tree_Dataset_SummerGT.zip`

If the dataset is stored elsewhere, update the relevant
path in the notebook.

### Step 2 - Open the Notebook

Open `MMA3001_Assignment.ipynb` in Google Colab.

Mount Google Drive when prompted and run the required
dependency installation cells.

### Step 3 - Load and Organise the Images

Run the initial notebook sections to:

1. Extract the dataset ZIP.
2. Identify RGB, depth and ground-truth files.
3. Match observations using their filenames.
4. Build the dataset metadata table.
5. Separate training and validation trees.

### Step 4 - Investigate the Depth Data

Run the exploratory sections that examine:

- Encoded depth values.
- Missing and zero-valued pixels.
- Nearest-neighbour interpolation.
- Linear interpolation.
- Sensitivity to missing patch size.

The encoded depth images do not have a verified physical
distance calibration.

Consequently, depth values are not interpreted as confirmed
measurements in millimetres or centimetres.

### Step 5 - Extract Tree Geometry

Use the ground-truth masks to calculate:

- Tree width and annotated area.
- Approximate trunk width.
- Branching position.
- Left and right branch spread.
- Branch asymmetry.

These features are measured in image pixels and relative
image coordinates.

### Step 6 - Train and Compare Initial Models

The notebook investigates:

- Dummy regression as a baseline.
- Random forest regression.
- Neural network regression.
- Global and spatial image features.

The models are compared using metrics such as mean absolute
error (MAE) and root mean squared error (RMSE).

### Step 7 - Build the Average Branch Skeleton

Ground-truth masks are analysed to identify approximate
branch centrelines.

The tree structures are aligned using their trunk positions,
then combined to estimate an average 2D branch geometry.

A straight-branch baseline is compared against the curved
average representation.

### Step 8 - Train the U-Net Segmentation Model

The U-Net neural network learns to predict tree structure
masks from image inputs.

Training is performed using summer observations and their
corresponding ground-truth masks.

The notebook separates fitting and development data for
model training and hyperparameter selection.

### Step 9 - Evaluate the Neural Network

Evaluate predicted masks against ground truth using:

- Dice coefficient.
- Intersection over Union (IoU).

Compare U-Net performance with the statistical average-mask
baseline.

A separate held-out group of trees is used for final
validation.

### Step 10 - Investigate Seasonal Transfer

Apply the summer-trained model to winter images.

Compare segmentation outputs across seasons and investigate
the contribution of RGB and encoded depth inputs.

Because winter masks are not independently annotated,
winter transfer is assessed qualitatively rather than as
a separate verified segmentation accuracy score.

### Step 11 - Generate the Final Average 2D Tree

Run the final average-tree code section after loading the
trained U-Net model.

The procedure:

1. Generates predicted tree-structure probability maps.
2. Aligns tree structures horizontally using trunk position.
3. Averages the aligned probability maps.
4. Visualises the representative 2D tree structure.
5. Saves the final results.

The model represents average projected tree geometry, not
a measurement of complete three-dimensional tree structure.

## 7. Validation Approach

Validation is designed to test whether the methods
generalise to previously unseen trees.

The original dataset split contains:

- 449 training trees.
- 113 held-out validation trees.

Separate development subsets are used during model training
and parameter selection.

Thresholds and hyperparameters should be selected using
development data, not the final held-out validation set.

Performance should be interpreted using quantitative metrics
alongside visual inspection of predicted masks.

## 8. Important Limitations

The following limitations should be considered when
interpreting the results:

- The final model is 2D rather than 3D.
- Physical tree dimensions cannot be recovered without
  suitable camera calibration or scale information.
- Encoded depth values do not have confirmed physical units.
- Ground-truth masks describe annotated tree structures,
  rather than every visible leaf or branch.
- Predictions may differ between summer and winter images.
- Averaging smooths differences between individual trees.
- A representative average shape does not capture every
  possible orchard tree geometry.

## 9. Reproducibility

For reproducible results:

- Use the provided dataset without changing filenames.
- Keep training and validation trees separate.
- Use the random seeds specified in the notebook.
- Fit scaling and preprocessing parameters using training
  data only.
- Preserve the trained model and evaluation outputs.
- Record any changes to parameters or preprocessing.
- Run notebook cells in order when starting a fresh session.

## 10. Final Output

The main deliverable is a representative average 2D tree
structure generated from neural network predictions.

This output can be used to investigate typical tree branching
geometry and to guide further research into machinery design.

The result is an image-based statistical representation and
should not be treated as a physically calibrated engineering
model without additional measurements.
