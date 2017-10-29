import caffe
import missinglink

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="485aee1a-7f13-0dab-c470-0be21d273407",
    project_token="KuqiSOcHQkzhavxl"
)

solver = missinglink_callback.create_wrapped_solver(
    caffe.SGDSolver,
    "poolmean_solver.prototxt"
)

missinglink_callback.set_properties(display_name="Videos 2 Language")

missinglink_callback.set_monitored_blobs(["softmax_loss", "accuracy"])

caffe.set_mode_cpu()

solver.solve()
