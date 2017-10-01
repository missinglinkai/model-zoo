# MissingLink.AI TensorFlow Model Zoo

In this folder you can find TensorFlow models integrated with the MissingLink.AI SDK.
The models are taken from the [TensorFlow model zoo](https://github.com/tensorflow/models) and from the [Google Cloud Platform cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples).
Every model comes with:
* An original `README` file, that contains explanation about the model, installation instructions and usage instructions;
* A `requirements.txt` file, that can be used with pip to install the required packages to run the model (see usage section for details).

## Usage

* Clone the repository.
* Choose the model you want to run.
* Install the required packages to run the model:
```
# from the model's directory
pip install -r requirements.txt
```
* Log in to the MissingLink.ai website and create a new project.
* In the file of the integrates model, find the following line:
```python
project = ML.TensorFlowProject(owner_id="your-owner-id", project_token="your-project-token")
```
OR
```python
project = ml.TensorFlowProject(owner_id="your-owner-id", project_token="your-project-token")
```
* Insert your owner ID and the project token in the appropriate places.
* Run the model.

## Main Integration Files

A list of all the integrated files.

* audioset
  - vggish_train_demo.py
* autoencoder
  - AdditiveGaussianNoiseAutoencoderRunner.py
  - MaskingNoiseAutoencoderRunner.py
  - AutoencoderRunner.py
  - VariationalAutoencoderRunner.py
* census
  - tensorflowcore/trainer/task.py
* compression-entropy_coder
  - core/entropy_coder_train.py
* inception
  - inception/inception_train.py
* learning_to_remember_rare_events
  - train.py
* lfads
  - run_lfads.py
* mnist
  - convolutional.py
* pcl_rl
  - trainer.py
