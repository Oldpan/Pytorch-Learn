import onnx
import tvm
import os
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
from tvm.contrib import graph_runtime as runtime
import numpy as np
import tvm.relay as relay
from PIL import Image
from tvm import rpc

onnx_model = onnx.load('../../test/new-mobilenetv2-128_S.onnx')

img = Image.open('../../datasets/hand-image/paper.jpg').resize((128, 128))

img = np.array(img).transpose((2, 0, 1)).astype('float32')
img = img / 255.0  # remember pytorch tensor is 0-1
x = img[np.newaxis, :]

input_name = '0'  # change '1' to '0'
shape_dict = {input_name: x.shape}
# sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

target = tvm.target.arm_cpu('rasp3b')

device_key = 'rpi3b'

network = 'new-mobilenetv2-128_S'
log_file = "%s,%s.log" % (device_key, network)
dtype = 'float32'


tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'n_trial': 1500,
    'early_stopping': 800,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder('default'),
        runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9090,
            number=5,
            timeout=10,
        )
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=False,
               try_winograd=True,
               try_spatial_pack_depthwise=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        tuner = 'xgb_knob'
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks[i] = tsk

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    input_shape = x.shape
    tasks = autotvm.task.extract_from_program(net, target=target,
                                            params=params,
                                            ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best recordsconnection reset
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, '192.168.1.104', 9090,
                                                timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


tune_and_evaluate(tuning_option)





# with relay.build_config(opt_level=3):
#     graph, lib, params = relay.build_module.build(sym, target, params=params)
#
# dtype = 'float32'
#
# temp = tempdir()
# path = temp.relpath('lib.tar')
# lib.export_library(path)
#
# host = '192.168.1.104'
# port = 9000
# remote = rpc.connect(host, port)
#
# remote.upload(path)
# rlib = remote.load_module('lib.tar')
#
# ctx = remote.cpu()
# module = runtime.create(graph, rlib, ctx)
# # set parameter (upload params to the remote device. This may take a while)
# module.set_input(**params)
# # set input data
# module.set_input('0', x)
# # run
# module.run()
# # get output
# out = module.get_output(0)
# # get top1 result
# top1 = np.argmax(out.asnumpy())
#
# print(top1)
