import os
import onnx
import torch
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
from onnxsim import simplify
from inpainting.base_network_inpainting import InpaintingGenerator
from serialization.pytorch_converter import convert
from serialization.utils import create_preprocess_dict, compress_and_save
from coremltools.models import MLModel
from PIL import Image


def serialize_model(checkpoint_path, activation="elu", gate_type_coarse="regular_conv", gate_type_fine="regular_conv", kernel_size=3):
    # Load the checkpoint
    folder = os.path.dirname(checkpoint_path)
    basename = os.path.splitext(os.path.splitext(checkpoint_path)[0])[0]
    netG = InpaintingGenerator(depth_factor=32, inference=False, activation=activation, gate_type_coarse=gate_type_coarse, gate_type_fine=gate_type_fine, kernel_size=kernel_size)
    netG_state_dict = torch.load(checkpoint_path)
    netG.load_state_dict(netG_state_dict)

    # Load the test input and output
    nn_img = torch.load('save/nn_img.pt') * 2. - 1.
    nn_msk = torch.load('save/nn_msk.pt')
    nn_output = torch.load('save/nn_output.pt') * 0.5 + 0.5

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
    ort_session = onnxruntime.InferenceSession(onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_inputs = {
            ort_session.get_inputs()[0].name: nn_img.detach().cpu().numpy(),
            ort_session.get_inputs()[1].name: nn_msk.detach().cpu().numpy(),
    }
    ort_out = ort_session.run(None, ort_inputs)[0][0] * 0.5 + 0.5
    ort_out = np.swapaxes(ort_out, 0, 2)
    ort_out = np.swapaxes(ort_out, 0, 1)
    plt.figure()
    plt.subplot(121)
    plt.imshow(nn_output[0].permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(ort_out)
    plt.show()

    # Serialization
    mlmodel = convert(onnx_path,
                      image_input_names=("img", "msk"),
                      minimum_ios_deployment_target='12',
                      scale={"img": np.array([2.]), "msk": np.array([1.])},
                      mean={"img": np.array([0.5]), "msk": np.array([0.])})
    pd = create_preprocess_dict(divisible_by=8,
                                resize_strategy='ByLongSide',
                                side_length=512,
                                output_classes='irrelevant')
    compress_and_save(mlmodel,
                      save_path=folder,
                      model_name=basename,
                      version=1.0,
                      ckpt_location=onnx_path,
                      preprocess_dict=pd,
                      model_description="Context Synthesis Inpainting",
                      convert_to_float16=True)

    # Check the CoreML model
    model = MLModel(os.path.join(folder, basename + ".mlmodel"))
    img = nn_img[0].permute(1, 2, 0).detach().numpy()
    img = Image.fromarray((255 * (img * 0.5 + 0.5)).astype(np.uint8), 'RGB')
    msk = nn_msk[0][0].detach().numpy()
    msk = Image.fromarray((255 * msk).astype(np.uint8), 'L')
    data = {"img": img, "msk": msk}
    ml_output = model.predict(data)["out"] * 0.5 + 0.5
    ml_output = np.swapaxes(ml_output, 0, 2)
    ml_output = np.swapaxes(ml_output, 0, 1)
    ml_output = np.swapaxes(ml_output, 0, 1)
    plt.figure()
    plt.subplot(121)
    plt.imshow(nn_output[0].permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(ml_output)
    plt.show()


checkpoint_path = "/Users/jchetboun/Projects/3d-photo-inpainting/checkpoints/nadav_inpainting_context_synthesis.pth.tar"
serialize_model(checkpoint_path, activation="elu", gate_type_coarse="regular_conv", gate_type_fine="regular_conv", kernel_size=3)
checkpoint_path = "/Users/jchetboun/Projects/3d-photo-inpainting/checkpoints/nadav_inpainting_context_synthesis_even_kernels_tanh.pth.tar"
serialize_model(checkpoint_path, activation="tanh", gate_type_coarse="regular_conv", gate_type_fine="regular_conv", kernel_size=4)
checkpoint_path = "/Users/jchetboun/Projects/3d-photo-inpainting/checkpoints/nadav_inpainting_context_synthesis_even_kernels_tanh_depth_separable.pth.tar"
serialize_model(checkpoint_path, activation="tanh", gate_type_coarse="depth_separable", gate_type_fine="depth_separable", kernel_size=4)
