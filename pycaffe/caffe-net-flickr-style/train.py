'''
This script should be run from the caffe root directory.
Please see instructions link for details!
'''

import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="485aee1a-7f13-0dab-c470-0be21d273407",
    project_token="KuqiSOcHQkzhavxl"
)

missinglink_callback.set_properties(display_name="Caffe Net Flickr Style")

solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, 'models/finetune_flickr_style/solver.prototxt')

missinglink_callback.set_expected_predictions_layers("label", "fc8_flickr")

solver.solve()
