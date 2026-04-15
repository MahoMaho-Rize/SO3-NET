import argparse
import os


def str2bool(x):
    return x.lower() in ("true")


parser = argparse.ArgumentParser("UprightNet")

# naming / file handling
parser.add_argument("--model_name", default="uprightnet", help="segmentation model")
parser.add_argument(
    "--data_dir", default="./datasets/uprightnet15/", help="point clouds dataset folder"
)
parser.add_argument(
    "--model_dir", default="./models/", help="training models log folder"
)
parser.add_argument(
    "--model_file", default="model.pth", help="pretrained network model file"
)
parser.add_argument("--gpu_idx", type=str, default="0,1,2")

# training parameters
parser.add_argument("--epoch", default=50, type=int, help="number of epoch in training")
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size in training"
)
parser.add_argument("--seed", type=int, default=-1, help="manual seed")
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="learning rate in training"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="weight decay (L2 penalty) of Adam optimizer",
)
parser.add_argument(
    "--decay_rate", type=float, default=0.01, help="decay rate of learning rate"
)
parser.add_argument("--no_decay", type=str2bool, default=False)

# model hyperparameters
parser.add_argument(
    "--alpha",
    default=0.1,
    type=float,
    help="coefficient of fitting residual loss function",
)
parser.add_argument("--num_points", type=int, default=2048)
parser.add_argument("--sym_op", type=str, default="max", help="symmetry operation")

parser.add_argument("--restore", action="store_true")

# ============================================================
# Equivariant network parameters (for --network equivariant)
# ============================================================
parser.add_argument(
    "--network",
    default="uprightnet",
    choices=["uprightnet", "equivariant"],
    help="network architecture: uprightnet (original) or equivariant (e3nn)",
)

# architecture
parser.add_argument(
    "--lmax",
    type=int,
    default=2,
    help="max spherical harmonic degree for edge features",
)
parser.add_argument(
    "--max_radius",
    type=float,
    default=0.5,
    help="radius for graph construction in equivariant network",
)
parser.add_argument(
    "--num_neighbors",
    type=float,
    default=32.0,
    help="number of KNN neighbors for graph construction",
)
parser.add_argument(
    "--irreps_hidden",
    type=str,
    default="128x0e+64x1o",
    help="hidden layer irreducible representations",
)
parser.add_argument(
    "--equi_layers",
    type=int,
    default=6,
    help="number of equivariant message passing layers",
)
parser.add_argument(
    "--radial_neurons", type=int, default=128, help="radial MLP hidden size"
)
parser.add_argument(
    "--conv_type",
    default="depthwise",
    choices=["fctp", "depthwise"],
    help="convolution type: fctp (full CG tensor product) or depthwise (MACE-style separable)",
)
parser.add_argument(
    "--num_radial_basis", type=int, default=16, help="number of radial basis functions"
)

# loss
parser.add_argument(
    "--loss_type",
    default="vmf",
    choices=["bce_fr", "geodesic", "vmf"],
    help="loss function type",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.1,
    help="auxiliary support BCE loss weight for equivariant model",
)
parser.add_argument(
    "--vmf_kappa_init",
    type=float,
    default=1.0,
    help="initial vMF concentration parameter",
)

# data
parser.add_argument(
    "--use_rotation_aug",
    type=str2bool,
    default=True,
    help="use rotation augmentation (set False for equivariant model)",
)
parser.add_argument(
    "--num_rotations",
    type=int,
    default=100,
    help="number of rotations per object: 100 (full), 10, 5, or 0 (no aug)",
)

opts = parser.parse_args()
