# The data science toolbox

This page presents the SEGES digital data science toolbox, which is an overview of the data science field, by grouping the sub-fields of data science and their respective methods and tools available as Python packages or programs. The figure below shows our take on visualizing data science with its sub-fields. The main gold for our team is to develop valuable software solutions or insightful technical articles or reports.

![Data science overview](https://github.com/PeterFogh/how_to_do_data_science/blob/master/data_science_overview.png)

<a href="https://app.diagrams.net/?mode=github#HPeterFogh%2Fhow_to_do_data_science%2Fmaster%2Fdata_science_overview.drawio" target="_blank">Edit</a>

## The data engineering toolbox

See topics of data management [on wikipedia here](https://en.m.wikipedia.org/wiki/Data_management).

- [Data integration](https://en.m.wikipedia.org/wiki/Data_integration)
	- ETL pipelines
- [Data provenance](https://en.m.wikipedia.org/wiki/Provenance#Data_provenance)
	- [Data lineage](https://en.m.wikipedia.org/wiki/Data_lineage)
- [Data preparation
	- Data cleaning
	- Abnormally detection
- [Data governance](https://en.m.wikipedia.org/wiki/Data_governance)
	- Data privacy:
		- Privacy via obfuscation:
			- De-identification:
				- Obfuscation of identifiers and quasi-identifiers via label encoding or cryptographic hashing (with "salt").
		- Privacy via generalization:
			- K-anonymization:
				- Generalization of attribute values, e.g. the values "cow" and "pig" are generalized to the values "animal".
				- Suppression of attributes. i.e. simply remove attributes that identify individual rows.
				- L-diversity of attribute values, such that each group of rows most contains L "well-represented" values.
				- T-closeness of attribute values, such that the value distribution of each group must be T-close to the distribution of the whole dataset.
		- Privacy via randomization
			- Differential privacy of aggregated attribute values, by injecting noise into the results such that result is the likelihood of an individual row affected the aggregated results is inconclusive.
- Metadata management and analytics

### Data types and data handling/manipulation in Python

-   Common [data structures in Python](https://docs.python.org/3/tutorial/datastructures.html): list, queue, tuple, set, dictionary, ordered dictionary.
-	Distributed sequential data - [dask.bag](https://docs.dask.org/en/latest/bag-overview.html)
-   Array data structure - [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) for data structure and [SciPy](http://scipy.github.io/devdocs/tutorial/general.html) for data operations
    -   Distributed array data structure- [dask.Array](http://docs.dask.org/en/latest/dataframe.html) which is based on NumPy for array data and [dask](https://docs.dask.org/en/latest/) for distributed operations.
    -   Tabular data structure (i.e. labelled index and columns) - [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) is based on NumPy for array data. Alternative, [vaex](https://github.com/vaexio/vaex) and [modin](https://github.com/modin-project/modin) supports dataframes for multi-threaded data operations, or [cuDF](https://github.com/rapidsai/cudf) supports dataframes stored and operated in GPU.
    -   Distributed tabular data structure - [dask.dataframe](http://docs.dask.org/en/latest/dataframe.html) which is based on pandas for tabular data and [dask](https://docs.dask.org/en/latest/) for distributed operations.
    -   Geospatial vector data structure - [geopandas.GeoDataFrame](http://geopandas.org/reference/geopandas.GeoDataFrame.html) which is based on pandas for tabular data and [Shapely](https://shapely.readthedocs.io/en/stable/manual.html) and [Fiona](https://fiona.readthedocs.io/en/stable/manual.html#data-model) for geospatial vector data structure and operations. All with base in [Geospatial Data Abstraction Library (GDAL](http://gdal.org/)). [cartopy](https://scitools.org.uk/cartopy/docs/latest/) simplifies the transformation of coordinate reference systems (CRSs). [PySAL](https://pysal.readthedocs.io/en/latest/) (Python Spatial Analysis Library) in an older package that facilitates tabular data like Pandas, but contains also tools for spatial analysis, e.g. special- weights, clustering, and regression.
    -   Geospatial raster data structure - [rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html) which is based on NumPy for array data and [GDAL](http://gdal.org/) for geospatial raster data structure and operations.
    -   (Distributed) multi-dimensional axis-labelled data structure - [xarray](http://xarray.pydata.org/en/stable/index.html) which is based on NumPy for array data, [dask](https://docs.dask.org/en/latest/) for distributed operations, and [NetCDF](https://en.wikipedia.org/wiki/NetCDF) for dataset model and storage format. [yt](https://yt-project.org/) is an alternative to [xarray](http://xarray.pydata.org/en/stable/index.html), however, it focuses on astrophysics and does not work with [dask](https://docs.dask.org/en/latest/).
-   Graph data structure - [networkx](https://networkx.github.io/) for data structure and data operations.
    -   Knowledge graphs - [Zincbase](https://github.com/tomgrek/zincbase) can build knowledge graphs by extracting triples and rules from unstructured text and query the graph. Wikipedia' knowledge graph is also public via [wikidata](https://www.wikidata.org).
-   Audio and music data structure - [LibROSA](https://librosa.github.io/librosa/index.html) for data structure, data operations, and data analysis.

## Data file formats

-   Tabular data: [CSV](https://en.wikipedia.org/wiki/Comma-separated_values), [TSV](https://en.wikipedia.org/wiki/Tab-separated_values), [XLS](https://en.wikipedia.org/wiki/Microsoft_Excel), [Parquet](https://en.wikipedia.org/wiki/Apache_Parquet).
-   Raster data [TIFF](https://en.wikipedia.org/wiki/TIFF), [PNG](https://en.wikipedia.org/wiki/Portable_Network_Graphics), [JPEG](https://en.wikipedia.org/wiki/JPEG).
-   Nested data: [XML](https://en.wikipedia.org/wiki/XML), [JSON](https://en.wikipedia.org/wiki/JSON).
-   Calendar data: [ICS](https://en.wikipedia.org/wiki/ICalendar).
-   Geospatial vector data: [Shapefile](https://en.wikipedia.org/wiki/Shapefile), [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON), [TAB](https://en.wikipedia.org/wiki/MapInfo_TAB_format), [GeoPackage](https://en.wikipedia.org/wiki/GeoPackage).
-   Geospatial raster data: [GeoTIFF](https://en.wikipedia.org/wiki/GeoTIFF), [GeoPackage](https://en.wikipedia.org/wiki/GeoPackage).
-   Multi-dimensional data: [NetCDF](https://en.wikipedia.org/wiki/NetCDF), [Zaar](https://zarr.readthedocs.io/en/stable/).

## Data annotation

-   Text annotation - [prodigy](https://prodi.gy/) (pay to use) - [LabelBox](https://labelbox.com/) (pay to use) - [doccano](https://github.com/chakki-works/doccano/blob/master/README.md)
-   Image annotation - [prodigy](https://prodi.gy/) (pay to use) - [Picterra](https://picterra.ch/) (pay to use) - [LabelBox](https://labelbox.com/) (pay to use) - [VIA](https://gitlab.com/vgg/via/) - [LabelImg](https://github.com/tzutalin/labelImg)
-   Video annotation - [VIA](https://gitlab.com/vgg/via/)
-   Audio annotation -  [VIA](https://gitlab.com/vgg/via/)

# The advanced analytics toolbox

## Statistics

-   Statistical measurements - Use the [SciPy](https://scipy.org/) Python package suite. Like [numpy.ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html), [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), or [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html) for data and all of or their statistical functions, e.g. [min](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.min.html), [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.mean.html), [max](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.max.html), [mode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html#scipy.stats.mode).
-   [Interpolation](https://en.wikipedia.org/wiki/Interpolation):
    -   Nearest, linear, cubic, spline interpolation - [scipy](https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)
    -   Kriging (also called Gaussian process regressions) - [pykriging](https://pykrige.readthedocs.io/en/latest/overview.html)
-   Outlier analysis:
    -   Mean +/- 3 x Standard Deviation - [numpy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
    -   Boxplot outlier via Interquartile Range (IQR) - [matplotlib](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html), [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html)
    -   Clustering outliers:
        -   DBSCAN outliers - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
            
        -   Isolation forest - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
            
        -   Robust random cut forest (RCF) - [rrcf](https://github.com/kLabUM/rrcf)
-   Correlation: Spearman (Spearman's rank correlation coefficient or Spearman's rho), Pearson (Pearson correlation coefficient (PCC) or the bivariate correlation - [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)  
    
-   Statistical test: t-test (Student's t-test), Wilcoxon signed-rank test, Chi-square test, Mann–Whitney U test, Analysis of variance (ANOVA) test - [scipy](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) - [statsmodels](https://www.statsmodels.org/stable/index.html) - [researchpy](https://github.com/researchpy/researchpy)
-   Curve fitting:
    -   Non-linear curve fitting - [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html), [lmfit](https://lmfit.github.io/lmfit-py/)
-   Spatial analysis: [PySAL](https://pysal.readthedocs.io/en/latest/)
    -   Spatial weights - [PySAL](https://pysal.readthedocs.io/en/latest/api.html#spatial-weights)
        -   Contiguity weights: Queen, Rook, Bishop
        -   Distance-based weights
            -   K-nearest neighbours weights
            -   Kernel W
    -   Spatial autocorrelation:
        -   Spatial lag - [PySAL](https://pysal.readthedocs.io/en/latest/api.html#spatial-lag)
        -   Moran statistics - [PySAL](https://pysal.readthedocs.io/en/latest/api.html#moran-statistics)
    -   Spatial clustering - [PySAL](https://pysal.readthedocs.io/en/latest/api.html#pysal-viz-mapclassify-choropleth-map-classification):
        -   Max-p-Regions - [PySAL](https://pysal.readthedocs.io/en/latest/generated/pysal.viz.mapclassify.Max_P_Classifier.html#pysal.viz.mapclassify.Max_P_Classifier) (uses [region](https://region.readthedocs.io/en/latest/users/max-p-regions/))
            
    -   Spatial regression - [PySAL](https://pysal.readthedocs.io/en/latest/api.html#pysal-model-linear-models-for-spatial-data-analysis):
        -   Spatial two-stage least squares (S2SLS) - [PySAL](https://pysal.readthedocs.io/en/latest/generated/pysal.model.spreg.GM_Lag.html#pysal.model.spreg.GM_Lag)

## Probability theory

See [Seeing Theory](https://seeing-theory.brown.edu/) for a good visual introduction basic probability, compound probability, probability distributions, frequentist inference, Bayesian inference, and regression analysis.

-   **TODO: Document probability theory concepts and Python packages.**

## Machine learning

-   ML frameworks:
    -   scikit-learn: **TODO: Document Python package and link to it.**
    -   gensim: **TODO: Document Python package and link to it.**
    -   tensorflow: **TODO: Document Python package and link to it.**
    -   pytorch: **TODO: Document Python package and link to it.**
    -   [cuML](https://github.com/rapidsai/cuml): offers traditional tabular ML tasks on GPUs without going into the details of CUDA programming. It shares compatible APIs with other  [RAPIDS](https://rapids.ai/)  projects.
    -   [NimbusML](https://github.com/microsoft/NimbusML) binding to [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet): TODO.
-   Feature engineering:
    -   Data vectorization (creating data embeddings):
        -   Categorical feature encoding - [feature-engine](https://feature-engine.readthedocs.io/en/latest/encoders/index.html)
        -   Text vectorization
            -   Word embeddings
                -   Word vectors - word2vec - [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
                -   Global Vectors (GloVe) - [spacy](https://spacy.io/api/vectors#from_glove) - [gensim](https://radimrehurek.com/gensim/models/keyedvectors.html#what-can-i-do-with-word-vectors)
                -   FastText model - [gensim](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText)
                -   VarEmbed Word Embeddings - [gensim](https://radimrehurek.com/gensim/models/wrappers/varembed.html#gensim.models.wrappers.varembed.VarEmbed)
            -   Document embeddings
                -   Term frequency (TF) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
                -   Term frequency-inverse document frequency (TF-IDF) - [gensim](https://radimrehurek.com/gensim/models/tfidfmodel.html) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
                -   Paragraph vectors - [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
        -   Graph vectorization:
            -   **TODO: document concepts and python packages.**
        -   Automated feature engineering - [featuretools](https://docs.featuretools.com/index.html)
    -   Feature selection:
        -   Recursive feature elimination (RFE) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
    -   Dimensionality reduction:
        -   Principal component analysis (PCA) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
        -   Linear discriminant analysis (LDA) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
        -   Singular-value decomposition (SVD) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
        -   T-distributed Stochastic Neighbor Embedding (t-SNE) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
    -   Missing data imputation - [feature-engine](https://feature-engine.readthedocs.io/en/latest/imputers/index.html)
    -   Numerical feature transformation - [feature-engine](https://feature-engine.readthedocs.io/en/latest/vartransformers/index.html)
    -   Feature discretisation - [feature-engine](https://feature-engine.readthedocs.io/en/latest/discretisers/index.html)
    -   Feature outlier capping - [feature-engine](https://feature-engine.readthedocs.io/en/latest/outliercappers/index.html)
    -   Feature augmentation - [Snorkel](https://github.com/snorkel-team/snorkel)
-   Active learning:
    -   **TODO: find python packages for active learning.**
-   Unsupervised learning:
    -   Clustering:
        -   Partitional clustering:
            -   k-means - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - [dask_ml](https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html#dask_ml.cluster.KMeans)
            -   k-medoids
        -   Hierarchical clustering:
            -   **TODO: document concepts and python packages.**
        -   Density-based clustering:
            -   Density-based spatial clustering of applications with noise (DBSCAN)- [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
    -   Topic modelling:
        -   Latent semantic analysis (LSA) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) (truncated SVD is know as LSA when used on term count/tf-idf matrices) - [gensim](https://radimrehurek.com/gensim/dist_lsi.html)
        -   Latent dirichlet allocation (LDA) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) - [gensim](https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore)
        -   lda2vec - [lda2vec](https://github.com/cemoody/lda2vec)
-   Supervised learning:
    -   Predictive models i.e. classification or regression) - see [dask_ml](https://dask-ml.readthedocs.io/en/latest/) for distributed computation of predictive models:
        -   Dummy (classifier [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html))
        -   K-nearest neighbours (k-NN) (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor))
        -   Linear models:
            -   Ordinary least squares (OLS) Linear (regressor [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html))
            -   Logistic regression (or logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier) (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))
            -   Ridge (regressor [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html))
            -   Lasso (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html))
            -   ElasticNet (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html))
            -   Perception (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)) is equivalent to SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None)
        -   Stochastic gradient descent (SGD) (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html))
        -   Gaussian process (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html))
        -   Naive Bayes (gaussian naive Bayes classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)) - (multinomial naive Bayes classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html))
        -   Support vector machine (SVM) (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html))
        -   Decision tree models:
            -   Decision tree (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html))
            -   Random forest (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))
            -   Boosted tree (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html))
            -   Extreme gradient boosting (XGBoost) - (classifier - [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)) / (regressor -  [xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor))
        -   Neural network models and deep learning models - see the [neural network zoo](http://www.asimovinstitute.org/neural-network-zoo/) - [tensorflow](https://www.tensorflow.org/) - [pytorch](https://pytorch.org/):
            -   Multi-layer perceptron (classifier - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)) / (regressor - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html))
            -   Convolutional neural network (CNN),
            -   Recurrent neural networks (RNNs):
                -   Recurrent neural networks (RNN)
                -   Long short-term memory (LSTM)
                -   Gated recurrent unit (GRU)
                -   Transformer
            -   Recursive neural network (RNN),
            -   Generative adversarial networks (GAN)
                -   pix2pix - [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
                -   CycleGAN - [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
            -   Deep residual networks (DRN)
    -   Transfor learning
        -   Model parameter initialization.
        -   Model sharing and repositories - [Pytorch Hub](https://pytorch.org/hub), [Papers With Code](https://paperswithcode.com/), [Tensorflow Models](https://github.com/tensorflow/models)
-   Quality assurance:
    -   Model validation and generalization ability, i.e. underfitting, fitting, or overfitting the test data
        -   Dataset sub-sampling for quicker analyses and training - [scipy.stats.kstest](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html) and [scipy.stats.chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) - see [this guide](https://medium.com/data-science-journal/how-to-correctly-select-a-sample-from-a-huge-dataset-in-machine-learning-24327650372c).
        -   Resampling strategies for imbalanced datasets - [imblearn](https://imbalanced-learn.org/en/stable/index.html)
            -   Simple random sampling
            -   Stratified Sampling
            -   Reservoir Sampling
            -   Random Undersampling and Oversampling
            -   Tomek Links Undersampling
            -   SMOTE (Synthetic Minority Oversampling Technique) oversampling
        -   Training, validation, and test dataset splitting - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - [dask_ml](https://ml.dask.org/modules/generated/dask_ml.model_selection.train_test_split.html)
        -   K-fold cross-validation - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
        -   Leave-One-Out cross-validation - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
    -   Hyperparameter optimizers - see the [sklearn function list](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers):
        -   Learning curve - for tunning the number of training samples - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
        -   Grid search - search hyperparameter space as a grid to find the local optimal setting in the grid - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [talos](https://github.com/autonomio/talos) (Keras)
        -   Random search - randomly search the hyperparameter space to find a local optimal setting - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    -   Feature importance:  
        -   Partial Dependence Plots - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.partial_dependence.plot_partial_dependence.html#sklearn.ensemble.partial_dependence.plot_partial_dependence) - [pdpbox](https://pdpbox.readthedocs.io/en/latest/pdp_plot.html)
        -   Individual Conditional Expectation - [PyCEbox](https://github.com/AustinRochford/PyCEbox)
        -   Permutation importance - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html) - [eli5](https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html#module-eli5.sklearn.permutation_importance)
        -   Local interpretable model-agnostic explanations (LIME) - [lime](https://github.com/marcotcr/lime)
        -   Shapley additive explanations (SNAP) - [shap](https://github.com/slundberg/shap)
    -   Performance compared to a group of humans on selected performance metrics.
    -   Pairwise distance metrics - see the [sklearn function list](https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics):
        -   Manhattan distance, i.e. L1 distance - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html)
        -   Euclidean distance, i.e. L2 distance - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)
        -   Cosine distance - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html)
        -   Hamming distance, also call edit distance - see [Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)
        -   Jaccard distance - see [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
    -   Clustering performance - see [sklearn](https://scikit-learn.org/stable/modules/classes.html#clustering-metrics)
        -   Silhouette score, i.e. cohesion and separation analysis - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score) - can be used for a silhouette analysis to select the optimal number of clusters, [see this example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
    -   Prediction performance:
        -   Classification performance metrics - see [sklearn](https://scikit-learn.org/stable/modules/classes.html#classification-metrics) and [Wikipedia about confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix):
            -   Confusion matrix - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)
                -   True positive (TP)
                -   True negative (TN)
                -   False positive (FP)
                -   False negative (FN)
            -   Recall also called sensitivity, hit rate, or true positive rate (TPR) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
            -   Precision also called positive predictive value (PPV) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
            -   F1-score - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
            -   Accuracy - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
            -   Hamming loss - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss)
            -   Jaccard similarity coefficient score - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html)
            -   Cohen’s kappa score - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
            -   Receiver Operating Characteristic Curve (ROC) - Note: only computable for binary classification - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
                -   Area Under the Curve (AUC) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
                -   Multiclass or multilabel performance metrics averaging: Micro, Macro, Weighted.
        -   Regularisation performance metrics - see [sklearn](https://scikit-learn.org/stable/modules/classes.html#regression-metrics):
    		-   Mean absolute error (MAE) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
            -   Mean squared error (MSE) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
            -   The coefficient of determination (_R_2) - [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
    -   Machine translation performance
        -   Bilingual Evaluation Understudy Score (BLEU) - [nltk](http://www.nltk.org/api/nltk.translate.html#module-nltk.translate.bleu_score)
    -   Fairness metrics and bias mitigation algorithms - [AIF360](https://github.com/IBM/AIF360)
    -   Adversarial attack and defence algorithms - [adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox)
-   Fairness:
    -   **TODO: Listen to ["Fairness in Machine Learning with Hanna Wallach" from "This Week in Machine Learning & Artificial Intelligence (AI) Podcast"(https://itunes.apple.com/dk/podcast/this-week-in-machine-learning-artificial-intelligence/id1116303051?l=da&mt=2&i=1000430267073).**
-   Auto ML
    -   Frameworks - [MLBox](https://github.com/AxeldeRomblay/MLBox), [Auto Sklean](https://github.com/automl/auto-sklearn), [TPOT](https://github.com/EpistasisLab/tpot),  [H20](http://docs.h2o.ai/h2o/latest-stable/), [Autokeras,](https://autokeras.com/) [Ludwig](https://uber.github.io/ludwig/)

# The Business intelligence toolbox

-   Visualization:
    -   Interactive tabular data, i.e. Dataframe - [qgrid](https://qgrid.readthedocs.io/en/latest/)
    -   DataFrame profiling - [edaviz](https://github.com/tkrabel/edaviz), [pandas-profilling](https://github.com/pandas-profiling/pandas-profiling)
    -   Data visualization - [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [plotly](https://plot.ly/python/), [PyViz](http://pyviz.org/) ([panel](https://panel.pyviz.org/), [hvplot](https://hvplot.pyviz.org/), [holoviews](http://holoviews.org/), [geoviews](http://geo.holoviews.org/), [datashader](http://datashader.org/), [boheh](https://bokeh.pydata.org/en/latest/), [colorcet](http://colorcet.pyviz.org/)), [altrair](https://altair-viz.github.io/), [yellowbrick](https://www.scikit-yb.org/en/latest/index.html)
        -   External memory (out-of-core) data visualization - [datashader](http://datashader.org/), [seaborn (KDE)](https://seaborn.pydata.org/generated/seaborn.kdeplot.html?highlight=kde#seaborn.kdeplot) 
    -   Geospatial visualization - [geoviews](http://geo.holoviews.org/), [kepler.gl](https://github.com/keplergl/kepler.gl/tree/master/bindings/kepler.gl-jupyter)
    -   3D visualization - [GemPy](https://github.com/cgre-aachen/gempy)
    -   Text data:
        -   Visualize Latent dirichlet allocation (LDA) and lda2vec with [pyldavis](https://pyldavis.readthedocs.io/en/latest/readme.html)
        -   Visualize text data general with [scattertext](https://github.com/JasonKessler/scattertext)
    -   Visualization web-services - [voila](https://github.com/QuantStack/voila)
    -   Machine learning and Deep learning visualization - [tensorwatch](https://github.com/microsoft/tensorwatch), [tensorboard](https://github.com/tensorflow/tensorboard), [visdom](https://github.com/facebookresearch/visdom)

# The software engineering toolbox

-   Development environment:  
    -   Integrated development environment (IDE):
        -   [VScode](https://code.visualstudio.com/) is a desktop application for code editing.
        -   [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) is a single webpage Jupyter notebook.
    -   Real-time collaboration IDE:
        -   [Colab](https://colab.research.google.com/)
        -   [CoCalc](https://cocalc.com/) - see [https://github.com/sagemathinc/cocalc-docker](https://github.com/sagemathinc/cocalc-docker)
-   Traditional software development:
    -   Graphical user interface (GUI) development - [wxPython](https://wxpython.org/), [Gooey](https://github.com/chriskiehl/Gooey), [Streamlit](https://streamlit.io/)
    -   Command-line interface (CLI) development:
        -   Progress bar - [tqdm](https://github.com/tqdm/tqdm), [alive-progress](https://github.com/rsalmei/alive-progress)
    -   Feature toggling - [ldclient](https://github.com/launchdarkly/python-server-sdk) (LaunchDarkly SDK for Python), [flipper](https://github.com/carta/flipper-client)
    -   Efficient Python via profilers:
        -   [`cProfile`](https://docs.python.org/3.6/library/profile.html#module-cProfile "cProfile") and [`profile`](https://docs.python.org/3.6/library/profile.html#module-profile "profile: Python source profiler.") provide deterministic CPU profiling of Python programs. It also has the IPython magic [%prun](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-prun).
        -   [line_profiler](https://github.com/rkern/line_profiler) is a module for doing line-by-line CPU profiling of functions. It also has the IPython magic [%lprun](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-prun).
        -   [memory_profiler](https://pypi.org/project/memory-profiler/) is a python module for monitoring memory consumption of a process as well as line-by-line analysis of memory consumption for python programs  It also has the IPython magics [%memit and %mprun](https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html#Profiling-Memory-Use:-%memit-and-%mprun)
        -   [ipython_memory_usage](https://github.com/ianozsvald/ipython_memory_usage) is an IPython tool to report memory usage deltas for every command you type.
        -   [SnakeViz](https://jiffyclub.github.io/snakeviz/) is a browser-based graphical viewer for the output of Python’s [cProfile](https://docs.python.org/3.4/library/profile.html#module-cProfile) module.
        -   [Py-spy](https://github.com/benfred/py-spy) is a sampling profiler for Python programs. It lets you visualize what your Python program is spending time on without restarting the program or modifying the code in any way.
        -   [psrecord](https://github.com/astrofrog/psrecord) is a small utility that uses the [psutil](https://github.com/giampaolo/psutil/) library to record the CPU and memory activity of a process.
        -   [Pympler](https://pythonhosted.org/Pympler/) is a development tool to measure, monitor and analyze the memory behavior of Python objects in a running Python application
        -   [Pyflame](https://github.com/uber/pyflame) high performance profiling tool that generates  [flame graphs](http://www.brendangregg.com/flamegraphs.html)  for Python.
    -   Software testing:
        -   Python script testing framework - [unittest](https://docs.python.org/3/library/unittest.html), [pytest](https://docs.pytest.org/en/latest/), [nose](https://nose.readthedocs.io/en/latest/), [hypothesis-auto](https://timothycrosley.github.io/hypothesis-auto/)
        -   Jupyter Notebook testing framework - [treon](https://github.com/ReviewNB/treon)
        -   Data mock-ups for testing:
            -   [pandas.util.testing](https://github.com/pandas-dev/pandas/blob/master/pandas/util/testing.py) has a lot of DataFrame mockups, e.g. [makeTimeDataFrame()](https://github.com/pandas-dev/pandas/blob/190a69e3d6bc3f106ef635ae18ff0fb8fdfe85c0/pandas/util/testing.py#L1707) or [makeMixedDataFrame()](https://github.com/pandas-dev/pandas/blob/190a69e3d6bc3f106ef635ae18ff0fb8fdfe85c0/pandas/util/testing.py#L1730).
    -   Debugging:
        -   Debuggers - [pdb](https://docs.python.org/3/library/pdb.html), [pdbpp](https://github.com/pdbpp/pdbpp/)
        -   Error messages - [stackprinter](https://github.com/cknd/stackprinter/)
    -   Code analysis: [flake8](http://flake8.pycqa.org/en/latest/), [dlint](https://github.com/duo-labs/dlint)
    -   Read domain-specific data formats:
        -   calendar data: for .ICS files use [ics](https://icspy.readthedocs.io)
-   Data science software development:
    -   Reproducible data and ML pipelines:
        -   [intake](https://intake.readthedocs.io/en/latest/quickstart.html) simplifies transferring and loading data from data sources and into Python.
        -   [DVC](https://dvc.org/doc) can version control large files (e.g. datasets and ML models), reproduce ML pipelines and track experiments.
        -   [MLflow](https://www.mlflow.org/docs/latest/index.html) can [track experiments](https://www.mlflow.org/docs/latest/tracking.html#tracking), [reproduce ML pipelines](https://www.mlflow.org/docs/latest/projects.html#projects), and [deploy ML models to production](https://www.mlflow.org/docs/latest/models.html#models).
        -   [ModelDB](https://github.com/mitdbg/modeldb) can track experiments.
        -   [Verta.ia](https://www.verta.ai/) uses ModelDB for tracking experiments, but can also deploy ML models to production, however it is not open source.
        -   [Kedro](https://kedro.readthedocs.io/en/latest/) is an alternative to MLflow and DVC, but with more focus on a common project folder structure.
        -   [Weight and Biases](https://www.wandb.com/) experiment tracking for deep learning, as it records and visualizes every detail of your research, collaborate easily, advance the state of the art.
