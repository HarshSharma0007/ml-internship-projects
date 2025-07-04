# import torch
# import matplotlib.pyplot as plt
# from captum.attr import IntegratedGradients
# from captum.attr import visualization as viz
# import os


# def get_attributions(model, input_tensor, target_class):
#     """
#     Compute attributions using Integrated Gradients.
#     """
#     model.eval()
#     ig = IntegratedGradients(model)
#     attributions = ig.attribute(
#         input_tensor, target=target_class, n_steps=200, internal_batch_size=32
#     )
#     return attributions


# def plot_attributions(attributions, original_image, target_label, save_path=None):
#     """
#     Visualize and optionally save attribution heatmap.
#     """
#     attr = attributions.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
#     orig = original_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

#     plt.figure(figsize=(6, 6))
#     viz.visualize_image_attr(
#         attr,
#         orig,
#         method="heat_map",
#         sign="positive",
#         show_colorbar=True,
#         title=f"Attribution: {target_label}"
#     )

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"✅ Attribution saved to {save_path}")

#     plt.close()

import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import os

def get_attributions(model, input_tensor, target_class):
    """
    Compute attributions using Integrated Gradients.
    """
    model.eval()
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        input_tensor, target=target_class, n_steps=200, internal_batch_size=32
    )
    return attributions

def plot_attributions(attributions, original_image, target_label, save_path=None):
    """
    Visualize and optionally save attribution heatmap.
    """
    attr = attributions.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    orig = original_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(6, 6))
    viz.visualize_image_attr(
        attr,
        orig,
        method="heat_map",
        sign="positive",
        show_colorbar=True,
        title=f"Attribution: {target_label}"
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Attribution saved to {save_path}")

    plt.close()
# End of file
