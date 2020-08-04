import os
import time
import onnx
import torch
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
from onnxsim import simplify
from inpainting.base_network_inpainting import InpaintingGenerator

epsilon = 1e-16


def evaluate_model(checkpoint_path, activation="elu", gate_type_coarse="regular_conv", gate_type_fine="regular_conv",
                   kernel_size=3):
    # Load the checkpoint
    folder = os.path.dirname(checkpoint_path)
    basename = os.path.splitext(os.path.splitext(checkpoint_path)[0])[0]
    netG = InpaintingGenerator(depth_factor=32, inference=False, activation=activation,
                               gate_type_coarse=gate_type_coarse, gate_type_fine=gate_type_fine,
                               kernel_size=kernel_size)
    netG_state_dict = torch.load(checkpoint_path)
    netG.load_state_dict(netG_state_dict)
    netG.to("cuda:0")
    netG.eval()

    # Load the test input and output
    nn_img = torch.load('save/nn_img.pt') * 2. - 1.
    nn_img = nn_img.to("cuda:0")
    nn_msk = torch.load('save/nn_msk.pt')
    nn_msk = nn_msk.to("cuda:0")

    # Export to ONNX
    onnx_path = os.path.join(folder, basename + ".onnx")
    torch.onnx.export(netG,
                      (nn_img, nn_msk),
                      onnx_path,
                      input_names=["img", "msk"],
                      output_names=["out", "fine", "coarse"],
                      opset_version=9)

    # Simplify
    onnx_model = onnx.load(onnx_path)
    model_simple, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_path = os.path.join(folder, basename + "_simple.onnx")
    onnx.save(model_simple, onnx_path)

    # Check the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    io_binding = ort_session.io_binding()
    io_binding.bind_input(name='img', device_type='cuda', device_id=0, element_type=np.float32, shape=nn_img.shape,
                          buffer_ptr=nn_img.data_ptr())
    io_binding.bind_input(name='msk', device_type='cuda', device_id=0, element_type=np.float32, shape=nn_msk.shape,
                          buffer_ptr=nn_img.data_ptr())
    io_binding.bind_output('out')
    io_binding.bind_output('fine')
    io_binding.bind_output('coarse')

    num_runs = 10
    tic = time.time()
    for i in range(num_runs):
        #         onnx_output = ort_session.run(None, ort_inputs)[0][0] * 0.5 + 0.5
        ort_session.run_with_iobinding(io_binding)
    toc = time.time() - tic
    onnx_output = io_binding.copy_outputs_to_cpu()[0]
    onnx_output = onnx_output[0] * 0.5 + 0.5

    print("Inference time ONNX", toc / num_runs)
    onnx_output = np.swapaxes(onnx_output, 0, 2)
    onnx_output = np.swapaxes(onnx_output, 0, 1)

    # Run Pytorch model
    tic = time.time()
    for i in range(num_runs):
        pytorch_output, _, _ = netG(nn_img, nn_msk)
    toc = time.time() - tic
    print("Inference time Pytorch", toc / num_runs)
    pytorch_output = pytorch_output[0].detach().cpu().numpy() * 0.5 + 0.5
    pytorch_output = np.swapaxes(pytorch_output, 0, 2)
    pytorch_output = np.swapaxes(pytorch_output, 0, 1)

    for i in range(10):
        arr1 = np.round(pytorch_output, decimals=i)
        arr2 = np.round(onnx_output, decimals=i)
        print("Comparison round", i, "decimals:", np.array_equal(arr1, arr2),
              np.count_nonzero(np.sum(np.abs(arr1 - arr2), axis=2)))

    pytorch_synthesis = np.zeros_like(pytorch_output)
    pytorch_synthesis[nn_msk[0][0].detach().cpu().numpy() > 0] = pytorch_output[nn_msk[0][0].detach().cpu().numpy() > 0]
    onnx_synthesis = np.zeros_like(onnx_output)
    onnx_synthesis[nn_msk[0][0].detach().cpu().numpy() > 0] = onnx_output[nn_msk[0][0].detach().cpu().numpy() > 0]
    print("Mean absolute difference:", np.mean(np.abs(pytorch_synthesis - onnx_synthesis)))
    print("Mean relative difference:",
          np.mean(np.abs((pytorch_synthesis - onnx_synthesis) / (pytorch_synthesis + epsilon))))

    plt.figure()
    ax = plt.subplot(121)
    plt.imshow(pytorch_output)
    plt.axis('off')
    ax.set_title("Pytorch")
    ax = plt.subplot(122)
    plt.imshow(onnx_output)
    plt.axis('off')
    ax.set_title("ONNX")
    plt.show()

    return


checkpoint_path = "/cnvrg/checkpoints/nadav_inpainting_context_synthesis.pth.tar"
evaluate_model(checkpoint_path, activation="elu", gate_type_coarse="regular_conv", gate_type_fine="regular_conv",
               kernel_size=3)
checkpoint_path = "/cnvrg/checkpoints/nadav_inpainting_context_synthesis_even_kernels_tanh_depth_separable.pth.tar"
evaluate_model(checkpoint_path, activation="tanh", gate_type_coarse="depth_separable", gate_type_fine="depth_separable",
               kernel_size=4)
