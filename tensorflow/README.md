# MissingLink.ai TensorFlow Model Zoo

In this folder you can find TensorFlow models integrated with the MissingLink.ai SDK.
The models are taken from the [TensorFlow model zoo](https://github.com/tensorflow/models) and from the [Google Cloud Platform cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples).
Every model comes with:
* An original `README` file, that contains explanation about the model, installation and usage instructions;
* A `README-ML` file, that contains details about the integration. Including:
  - The origin of the model (TF models OR Google Cloud samples)
  - Which files were integrated.
  - The expected result from running the model (?)
* A `requirements.txt` file, that can be used with pip to install the required packages to run the model:
```
pip install -r requirements.txt
```
