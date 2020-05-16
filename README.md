# BioMRC_code
BIOMRC Paper Preprint: https://arxiv.org/abs/2005.06376
(To be presented in BioNLP 2020)

Abstract:
We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the new dataset, and that two neural MRC models  that had been tested on BIOREAD perform much better on BIOMRC, indicating that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better.

Dataset is available on: https://archive.org/details/biomrc_dataset

The ASReaderTest and AOAReaderTest files use as input a preprocessed version of the dataset, which is not provided, but can be created from the original dataset and saved in the same format as a list of tokens instead of text of the abstracts, titles etc.

Example of preprocessing (Setting A):

>Abstract: ['single-agent', 'activity', 'for', 'entity8253', ... , 'hormonal', 'or', 'single-agent', 'entity4349', 'therapy']
>
>Title: ['no', 'synergistic', 'activity', 'of', 'entity1259', 'and', 'xxxx', 'in', 'the', 'treatment', 'of', 'entity157']
>
>Entities List: ['entity1', 'entity632', 'entity137', 'entity440', 'entity8253', 'entity4349', 'entity5', 'entity1259', 'entity2262', 'entity4020', 'entity157', 'entity221', 'entity1851']
>
>Answer: 9

You can use your own tokenization/preprocessing functions to achieve the above format, so that the code can run, or you can implement/tweak the code to run with your own format.

Setting B can be enabled as a parameter and the code will perform the mapping from the Setting A version of the preprocessed dataset.

The code SciBertReaderSum and SciBertReaderMax uses the original dataset and not a preprocessed one, as it relies on the tokenization to Sentence Piece Tokens by BERT.

For Setting B in SciBertReaderSum and SciBertReaderMax, just change the dataset input to the B version.
