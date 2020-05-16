# BioMRC_code
BIOMRC Paper Preprint: https://arxiv.org/abs/2005.06376
(To be presented in BioNLP 2020)

Abstract:
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset, and that two neural MRC models  that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better.

Dataset is available on: https://archive.org/details/biomrc_dataset

The ASReaderTest and AOAReaderTest files use as input a preprocessed version of the dataset, which is not provided, but can be created from the original dataset and saved in the same format (as a list of tokens instead of a text string).

Setting B can be enabled as a parameter and the code will perform the mapping from the Setting A version of the preprocessed dataset.

The code SciBertReaderSum and SciBertReaderMax uses the original dataset and not a preprocessed one.

For Setting B in SciBertReaderSum and SciBertReaderMax, just change the dataset input to the B version.
