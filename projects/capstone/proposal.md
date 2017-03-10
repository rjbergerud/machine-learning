## Proposal for Captstone Project
### MLND
Ryder Bergerud, Victoria BC

Dataset available on [UCI][3], and was once a [Kaggle-competition][8]
#### Domain Background
This project attempts to assess the trade-offs between modeling forest-cover based on cartographic data and a combination of cartographic data and satellite-imagery data.  There is much less data to process with cartographic variables, and so both many-more points might be trained on compared with to satellite-imagery given the same resources.

The techniques and data in this project could be useful for assessing human-management of forests.  For instance, seeing if a a machine could distinguish between forests at different stages of growth in replanting, or infer other characteristics of the forest based on similar features.

There are many uses for mapping forest cover.  Similar features have combined supervised learning with [mapping fungi] [1].  Different types of underground forest fungi were characterized by different spectral patterns of trees from moderate-resolution images.  The model was originally trained on a labeled dataset of plots, where data was infered from trees that supported only on or the other type of mycorizhal fungi.  Tracking forest-loss and harvesting is done on logging, as on [globalforestwatch][2].

<!-- http://www.sciencedirect.com/science/article/pii/S0304380005005740?np=y&npKey=d2f9a6b25bcf8da91640b59900f15908da6da2143e7a1b9e9d64ff43881f6f2a -->
Similar techniques and models could be useful for tracking changes in forest cover associated with climate change, quantifying large-scale stress due to drought on forests, or tracking pine-beetle spread.

A suitable application of the following model would be to assess whether forests that have been harvested are being re-stocked with the appropriate species.

[1]: https://www.nasa.gov/feature/jpl/nasa-satellite-images-uncover-underground-forest-fungi  "NASA Satellite Images Uncover Underground Forest Fungi"
[2]: globalforestwatch.org
[8]: https://www.kaggle.com/c/forest-cover-type-prediction "kaggle"

#### Problem Statement
To be able to predict the forest type of areas that can be surveyed on-the-ground, researchers must rely on cartographic data (location, slope), publicly-available geological data, and satellite imagery.  The USFS released this data so that they may infer the forest-type of adjacent lands to National Forests, as well as inholdings.

#### Dataset and inputs
We will use the data from the UCI Dataset repository, [covertype][3].  There are 12 features and 1 target variable, cover_type.   As stated in the description on the UCI repository:

>This study area includes four wilderness areas located in the
Roosevelt National Forest of northern Colorado.  These areas
represent forests with minimal human-caused disturbances,
so that existing forest cover types are more a result of
ecological processes rather than forest management practices.

Additional data from landsat images will be collected through the [nasa api][4].  The data will be resized to represent the 30x30m area that is considered in the cartographic dataset for each point.  This will represent approximated 3x3px, using the conversion that 1 degree latitude is equivalent  to a distance of 111km.  Data will be chosen from the fall season, when differences is color-characteristics of trees will likely be greatest, at least between coniferous and non-coniferous species.  If there are resources to accommodate it, it is also possible that images might be chosen from different seasons to improve accuracy.

I could use semi-supervised methods, mining new data that is unlabeled, label it using unsupervised learning methods, and then training on that.  However, we have more than enough data to train on.

Note that given the size of this dataset (over 500 000 points) its possible that it was generated using unsupervised learning methods, and hence the training data may display some bias may not be fully accurate.  I haven't been able to find a reference to the collection of the data on the US Forest Service site, so this remains unverified.

Additionally, some of the ground-data might accommodate multiple labels (mixed-type forests), as discussed in  the paper [mapping fungi] [1].

Also considered was using the [USFS Forest Inventory Assessment Data][7] (FIA).  However, much of the data available labeling forest cover type was mostly only complete for stocked forests.  This might be suitable for training on since forest type of stocked forests are largely due to forest-management practices, as opposed to natural features.   See [Database Description and Users Guide][5], p56.  Since the data is not as well-prepared as the UCI dataset, a suitable continuation of the project would be to apply the more successful methods to the subset of original-growth forest in the FIA dataset.

<!--
Since National Core Field Guide, Version 2.0, 2004
The crew determined the
estimated forest type by either recording the previous forest type on
remeasured plots or, on all other plots, the most appropriate forest type to the
condition based on the seedlings present or the forest type of the adjacent
forest stands -->

<!-- -Add data from Nasa landsat-8.
  - Sample image is 134kb.  Could compress and transform on download.
  - http://www.csgnetwork.com/degreelenllavcalc.html
  - 1 degree latitude at 39 degrees (roosevelt forest is here) 111015.45481323975 metres, same for 1 degree longitude generally
  - Each image represents 0.025 degrees, so 0.025*111 = 2.775 km.
  - To represent an image 0.03m in length, we would take about 1/100th the scale, so 0.00025 of a degree.  This corresponds to a 3x3px segment of the 291x291px image.  Color this means 4*(3*3)= 36 dimensions.
  - Otherwise, data will be 130*150000/1000 = 19 GB
  - https://api.nasa.gov/api.html#earth
-Data is a mix of continuous data such as Elevation, aspect, slope, distance to water-features, and hillshade, along with categorical variables Wilderness area-designation, soil-type.
-Output is one of seven cover-types. -->

[3]: https://archive.ics.uci.edu/ml/datasets/Covertype "covertype".
[4]: https://api.nasa.gov/ "nasa.gov"
[5]: https://www.arb.ca.gov/regact/2014/capandtradeprf14/4_FIADB_user%20guide_6-0p2_5-6-2014.pdf "The Forest Inventory and Analysis Database Guide"
[7] https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html "FIA DataMart"
#### Solution Statement
Given data from cartographic variables only, and then comparing and combining with data from landsat satellite images, use a labeled dataset to train identification one of 7 types of forest cover in the Roosevelt National forest.


#### Benchmark Model
- Can compute accuracy score of just assuming 1-type on all of data.

https://pdfs.semanticscholar.org/42fd/f2999c46babe535974e14375fbb224445757.pdf
ANN gives 70.58 percent accuracy.

https://www.kaggle.com/sharmasanthosh/forest-cover-type-prediction/exploratory-study-of-ml-algorithms/notebook
ExtraTrees bagging method produces 88% accuracy (.90/.10 size test/train split)

Exists model on kaggle with 100% accuracy.  Might be from cheating, non of the top ten have released their models.

Since their are many classes, should use confusion matrix to identify which classes are most difficult to classify.


#### Evaluation Metrics
To evaluate the model, I will use multi-class classification accuracy.  The  Kaggle competition that used this dataset used the same metric to score and rank its contestants.  

$$ \frac{\sum_{x \in X} 1(h(x) = x)}{|X|} $$

where $$1(h(x) = x)$$ is 1 if our estimator's prediction $$h(x)$$ equals the ground-truth $$x$$.  This is a special case of the Jacard similarity score,

$$ J(A,B) = \frac{A \cap B}{A \cup B} $$

where $$A = 1(h(x) = x)$$, and $$B$$ is the sample $$X$$.

In other situations, where certain trees were over-represented in our sample set, we might consider using a score that scores each class, and then weights them equally.  An example of this would be the precision score, with averaging set to macro.  However, this is not the situation, as each tree-type is well represented in the data.

#### Project Design

##### Visualization:
Results pre-processing steps should be chosen in light of what is observed in the following visualizations and data-explorations.  For instance, If many features seem highly-correlated, there might be some dependency between them that can be modeled with independent component analysis.

Plot on a map which coordinates are represented by the data.  They should represent four different plot areas according to the information provided with the data.  Note if there are any irregularities.

Include the following visualizations as well:
-Map plot of tree-types, to see how much they cluster by coordinates
-Correlation matrix scatter plot
-Pick an area of several different cover-types, and impose labels over top a satellite image.

##### Pre-processing:
Pre-processing steps will likely include feature-selection, given the number of features in the dataset (over 50), and the number of training-points.  Our options include PCA (principle component analysis), ICA, and feature-selection from a model.  Dimension reduction might be considered separately on the landsat images data, which will have have 4*9=36 dimensions given by the 4 color channels.  

The process from here on in will be to (A) train the model just on cartographic variables, (b) train it on the landsat images, and (c) use both features sets to train a final model.  Comparing accuracy scores and checking *feature_importances_* will inform how useful the landsat images are compared the cartographic variables in classification.

Additionally, features scaling will likely have to be considered depending on the model used below.  

##### Modelling:
Using k-fold cross validation with parameter tuning (likely grid-search), I will try the following models based on their compatibility with both categorical and continuous data:
 - Decision tree
 - ExtraTrees
 - Artificial Neural Network
 - LogisticRegression


<!-- Converted to html via https://upmath.me/ -->
