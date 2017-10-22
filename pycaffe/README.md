# MissingLink.AI pyCaffe Model Zoo
In this folder you can find pyCaffe models integrated with the MissingLink.AI SDK. The models are taken from the following sources:

## Sources

* [Age Gender Deep Learning](https://github.com/GilLevi/AgeGenderDeepLearning)
* [Data Augumentation Testing](https://github.com/gombru/dataAugmentationTesting)
* [Network In Network](https://gist.github.com/mavenlin/e56253735ef32c3c296d)
* Caffe Net On Flickr Style: [source](https://github.com/BVLC/caffe/tree/master/models/finetune_flickr_style), [instructions](https://github.com/BVLC/caffe/tree/master/examples/finetune_flickr_style)

## Usage

* Clone the repository.
* Choose the model you want to run.
* Find the following lines in the integrated file:
```python
missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
```
* Insert your owner ID and your project token in the appropriate places. 
* Run the model.

**Note:** Some models include a `README` with more instructions required to run them.

## Main Integration Files

* [Age Gender Deep Learning](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/age-gender-deep-learning)
  * age-gender-deep-learning/age_net_definitions/train.py
  * age-gender-deep-learning/gender_net_definitions/train.py
* [Caffe Net Flickr Style](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/caffe-net-flickr-style/train.py)
  * caffe-net-flickr-style/train.py
* [Data Augmentation Testing](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/data-augmentation-testing/train.py)
  * data-augmentation-testing/train.py
* [Network In Network](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/network-in-network/train.py)
  * network-in-network/train.py
