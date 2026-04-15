import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from network import UprightNet
import numpy as np
from sklearn import linear_model
import random
from Common import loss_utils
from Common.RobustPointSetDataLoader import RobustPointSetDataLoader

# Equivariant model imports (lazy: only loaded when --network equivariant)
_equivariant_imports_loaded = False


def _load_equivariant_imports():
    global _equivariant_imports_loaded
    if _equivariant_imports_loaded:
        return
    global EquivariantPointCloudDataset, build_equivariant_model
    global EquivariantLoss, angular_error_deg, rotation_matrix_to_upright
    global PyGDataLoader
    from Common.EquivariantDataLoader import EquivariantPointCloudDataset
    from network_equivariant import build_equivariant_model
    from Common.loss_equivariant import EquivariantLoss
    from Common.geometric_utils import angular_error_deg, rotation_matrix_to_upright
    from torch_geometric.loader import DataLoader as PyGDataLoader

    _equivariant_imports_loaded = True


torch.cuda.current_device()
torch.cuda._initialized = True


class Model:
    def __init__(self, opts):
        self.opts = opts

    def train(self):
        """DATA LOADING"""
        print("Loading dataset ...")
        trainDataLoader = DataLoader(
            RobustPointSetDataLoader(self.opts, partition="train"),
            batch_size=self.opts.batch_size,
            shuffle=True,
        )
        testDataLoader = DataLoader(
            RobustPointSetDataLoader(self.opts, partition="test"),
            batch_size=self.opts.batch_size,
            shuffle=False,
        )

        print("The number of training data is: %d" % len(trainDataLoader.dataset))
        print("The number of testing data is: %d" % len(testDataLoader.dataset))

        """MODEL LOADING"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts.gpu_idx
        # Create model
        if self.opts.model_name == "uprightnet":
            segmentor = UprightNet().cuda()
        else:
            raise ValueError("Unknown network: %s" % (self.opts.model_name))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            segmentor = nn.DataParallel(segmentor)

        # Set random seed for reproducibility
        if self.opts.seed < 0:
            self.opts.seed = random.randint(1, 10000)
        print("Random Seed: %d" % (self.opts.seed))
        random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)

        # Create optimizer
        optimizer = torch.optim.Adam(
            segmentor.parameters(),
            lr=self.opts.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.weight_decay,
        )
        if self.opts.no_decay:
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=self.opts.decay_rate
            )

        """TRAINING"""
        print("Start training...")

        for epoch in range(self.opts.epoch):
            print("Epoch %d / %s:" % (epoch + 1, self.opts.epoch))
            if scheduler is not None:
                scheduler.step(epoch)

            train_loss_bce = 0.0
            train_loss_fr = 0.0

            for _, data in tqdm(
                enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
            ):
                points_original, points_rotation, _, _, pid, coef_d = data
                points_original, points_rotation, pid, coef_d = (
                    points_original.cuda(),
                    points_rotation.cuda(),
                    pid.cuda(),
                    coef_d.cuda(),
                )

                points_rotation = points_rotation.transpose(2, 1).contiguous()
                segmentor = segmentor.train()
                pred = segmentor(points_rotation)
                pred = pred.squeeze()

                optimizer.zero_grad()
                bce_loss = loss_utils.bce_loss(pred, pid)
                fr_loss = loss_utils.fr_loss(pred, points_original, coef_d)
                loss = bce_loss + self.opts.alpha * fr_loss
                train_loss_bce += bce_loss * self.opts.batch_size
                train_loss_fr += fr_loss * self.opts.batch_size
                loss.backward()
                optimizer.step()
            print("BCE Loss: %.3f, FR Loss: %.3f" % (train_loss_bce, train_loss_fr))

        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        torch.save(
            segmentor.state_dict(),
            os.path.join(self.opts.model_dir, "model" + current_time + ".pth"),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(self.opts.model_dir, "optimizer" + current_time + ".pth"),
        )

        print("End of training...")

        """TESTING"""
        print("Start testing...")
        orientations = torch.empty((0, 3))
        for _, data in tqdm(
            enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9
        ):
            points_original, points_rotation, _, rotm, _, _ = data
            points_original, points_rotation, rotm = (
                points_original.cuda(),
                points_rotation.cuda(),
                rotm.cuda(),
            )

            points_rotation = points_rotation.transpose(2, 1).contiguous()
            segmentor = segmentor.eval()
            with torch.no_grad():
                test_pred = segmentor(points_rotation)

            orientation = self.UprightOriEst(test_pred.squeeze(), points_original, rotm)
            orientations = torch.cat((orientations, orientation), dim=0)

        angle = torch.acos_(orientations[:, 1]) / torch.pi * 180
        me = torch.mean(angle)
        acc = torch.sum(angle < 10) / angle.shape[0] * 100
        print("Mean Error:", me)
        print("Accuracy:", acc)
        print("End of testing...")

    def test(self):
        """DATA LOADING"""
        print("Loading dataset ...")
        testDataLoader = DataLoader(
            RobustPointSetDataLoader(self.opts, partition="test"),
            batch_size=self.opts.batch_size,
            shuffle=False,
        )
        print("The number of testing data is: %d" % len(testDataLoader.dataset))

        """MODEL LOADING"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts.gpu_idx
        # Create model
        if self.opts.model_name == "uprightnet":
            segmentor = UprightNet().cuda()
        else:
            raise ValueError("Unknown network: %s" % (self.opts.model_name))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            segmentor = nn.DataParallel(segmentor)
        model_file = os.path.join(self.opts.model_dir, self.opts.model_file)
        segmentor.load_state_dict(torch.load(model_file))

        # Set random seed for reproducibility
        if self.opts.seed < 0:
            self.opts.seed = random.randint(1, 10000)
        print("Random Seed: %d" % (self.opts.seed))
        random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)

        """TESTING"""
        print("Start testing...")
        orientations = torch.empty((0, 3))
        for _, data in tqdm(
            enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9
        ):
            points_original, points_rotation, _, rotm, _, _ = data
            points_original, points_rotation, rotm = (
                points_original.cuda(),
                points_rotation.cuda(),
                rotm.cuda(),
            )

            points_rotation = points_rotation.transpose(2, 1).contiguous()
            segmentor = segmentor.eval()
            with torch.no_grad():
                test_pred = segmentor(points_rotation)

            orientation = self.UprightOriEst(test_pred.squeeze(), points_original, rotm)
            orientations = torch.cat((orientations, orientation), dim=0)

        angle = torch.acos_(orientations[:, 1]) / torch.pi * 180
        me = torch.mean(angle)
        acc = torch.sum(angle < 10) / angle.shape[0] * 100
        print("Mean Error:", me)
        print("Accuracy:", acc)
        print("End of testing...")

    def UprightOriEst(self, pred, original, rotm):
        bsize = original.size()[0]
        pred = pred > 0.5
        orientations = torch.empty([0, 3])
        ransac = linear_model.RANSACRegressor(residual_threshold=0.03)
        for i in range(bsize):
            points = original[i]
            mcenter = torch.mean(points, axis=0).cpu()
            spoints = torch.index_select(
                points, dim=0, index=pred[i, :].nonzero().squeeze()
            ).cpu()  # supporting points
            numspoints = spoints.shape[0]
            # the normal of supporting plane pointing to the mass center
            if numspoints >= 3:
                spoints = spoints
                ransac.fit(spoints[:, [0, 2]], spoints[:, 1])
                a, c = ransac.estimator_.coef_  # coefficients
                d = ransac.estimator_.intercept_  # intercept
                orientation = (
                    torch.tensor([a, -1.0, c])
                    / torch.norm(torch.tensor([a, -1.0, c]), p=2)
                    * torch.sign(a * mcenter[0] - 1.0 * mcenter[1] + c * mcenter[2] + d)
                )
            # the unit vector from center of supporting points to the mass center
            elif numspoints > 0:
                scenter = torch.mean(spoints, axis=0)
                orientation = (mcenter - scenter) / torch.norm(mcenter - scenter, p=2)
            # the fixed output [0, 1, 0] (we use the inversed rotation matrix as output if estimate the original point cloud)
            else:
                # orientation = torch.tensor([0., 1., 0.], dtype=torch.float64)
                orientation = torch.inverse(rotm[i].cpu())[1]
            orientations = torch.cat((orientations, orientation.unsqueeze(0)), dim=0)
        return orientations

    # ================================================================
    # Equivariant model methods (e3nn-based)
    # ================================================================

    def train_equivariant(self):
        """Train the E(3)-equivariant upright orientation network."""
        _load_equivariant_imports()

        """DATA LOADING"""
        print("Loading dataset ...")
        train_dataset = EquivariantPointCloudDataset(self.opts, partition="train")
        test_dataset = EquivariantPointCloudDataset(self.opts, partition="test")
        trainDataLoader = PyGDataLoader(
            train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        testDataLoader = PyGDataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        print("The number of training data is: %d" % len(train_dataset))
        print("The number of testing data is: %d" % len(test_dataset))

        """MODEL LOADING"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts.gpu_idx
        model = build_equivariant_model(self.opts).cuda()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model parameters: %d" % num_params)

        # Set random seed
        if self.opts.seed < 0:
            self.opts.seed = random.randint(1, 10000)
        print("Random Seed: %d" % self.opts.seed)
        random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.opts.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.weight_decay,
        )
        if self.opts.no_decay:
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=self.opts.decay_rate,
            )

        # Loss function
        criterion = EquivariantLoss(
            loss_type=self.opts.loss_type,
            beta=self.opts.beta,
            antipodal=True,
        )

        """TRAINING"""
        print("Start equivariant training...")
        print("  Loss type: %s" % self.opts.loss_type)
        print("  Rotation augmentation: %s" % self.opts.use_rotation_aug)
        print("  Hidden irreps: %s" % self.opts.irreps_hidden)
        print("  Layers: %d, lmax: %d" % (self.opts.equi_layers, self.opts.lmax))

        best_me = float("inf")

        for epoch in range(self.opts.epoch):
            print("Epoch %d / %s:" % (epoch + 1, self.opts.epoch))
            if scheduler is not None:
                scheduler.step(epoch)

            model.train()
            epoch_loss_dir = 0.0
            epoch_loss_sup = 0.0
            epoch_loss_total = 0.0
            epoch_kappa = 0.0
            num_batches = 0

            for data in tqdm(
                trainDataLoader, total=len(trainDataLoader), smoothing=0.9
            ):
                data = data.cuda()

                # Forward pass
                outputs = model(data)

                # Prepare targets
                targets = {
                    "y_direction": data.y_direction,  # (B, 3)
                    "y_support": data.y_support,  # (N,)
                }

                # Loss
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                epoch_loss_total += loss_dict["total"].item()
                epoch_loss_dir += loss_dict["direction"].item()
                epoch_loss_sup += loss_dict["support"].item()
                epoch_kappa += loss_dict["kappa_mean"].item()
                num_batches += 1

            avg_total = epoch_loss_total / num_batches
            avg_dir = epoch_loss_dir / num_batches
            avg_sup = epoch_loss_sup / num_batches
            avg_kappa = epoch_kappa / num_batches
            print(
                "  Loss: %.4f (dir: %.4f, sup: %.4f), kappa: %.2f"
                % (avg_total, avg_dir, avg_sup, avg_kappa)
            )

            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == self.opts.epoch - 1:
                me, acc5, acc10 = self._evaluate_equivariant(model, testDataLoader)
                print(
                    "  [Eval] Mean Error: %.2f deg, Acc@5: %.1f%%, Acc@10: %.1f%%"
                    % (me, acc5, acc10)
                )
                if me < best_me:
                    best_me = me
                    current_time = datetime.now().strftime("%Y%m%d-%H%M")
                    save_path = os.path.join(
                        self.opts.model_dir,
                        "equivariant_best_%s.pth" % current_time,
                    )
                    torch.save(model.state_dict(), save_path)
                    print("  [Saved] Best model (ME=%.2f) to %s" % (best_me, save_path))

        # Save final model
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        torch.save(
            model.state_dict(),
            os.path.join(
                self.opts.model_dir, "equivariant_final_%s.pth" % current_time
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                self.opts.model_dir, "equivariant_optimizer_%s.pth" % current_time
            ),
        )
        print("End of equivariant training. Best Mean Error: %.2f deg" % best_me)

    def test_equivariant(self):
        """Test the E(3)-equivariant upright orientation network."""
        _load_equivariant_imports()

        """DATA LOADING"""
        print("Loading dataset ...")
        test_dataset = EquivariantPointCloudDataset(self.opts, partition="test")
        testDataLoader = PyGDataLoader(
            test_dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        print("The number of testing data is: %d" % len(test_dataset))

        """MODEL LOADING"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opts.gpu_idx
        model = build_equivariant_model(self.opts).cuda()

        model_file = os.path.join(self.opts.model_dir, self.opts.model_file)
        model.load_state_dict(torch.load(model_file))
        print("Loaded model from %s" % model_file)

        # Set random seed
        if self.opts.seed < 0:
            self.opts.seed = random.randint(1, 10000)
        print("Random Seed: %d" % self.opts.seed)
        random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)

        """TESTING"""
        me, acc5, acc10 = self._evaluate_equivariant(
            model, testDataLoader, verbose=True
        )
        print("=" * 50)
        print("Mean Angular Error: %.2f deg" % me)
        print("Accuracy@5:  %.2f%%" % acc5)
        print("Accuracy@10: %.2f%%" % acc10)
        print("=" * 50)

    def _evaluate_equivariant(self, model, dataloader, verbose=False):
        """
        Evaluate equivariant model on a dataloader.

        Returns:
            mean_error: mean angular error in degrees
            acc5:       percentage of predictions within 5 degrees
            acc10:      percentage of predictions within 10 degrees
        """
        _load_equivariant_imports()
        model.eval()

        all_errors = []
        all_kappas = []

        with torch.no_grad():
            for data in tqdm(
                dataloader, total=len(dataloader), smoothing=0.9, disable=not verbose
            ):
                data = data.cuda()
                outputs = model(data)

                # Predicted direction
                mu = outputs["direction_mu"]  # (B, 3)
                kappa = outputs["direction_kappa"]  # (B, 1)

                # Ground truth direction
                gt_direction = data.y_direction  # (B, 3)

                # Angular error
                errors = angular_error_deg(mu, gt_direction, antipodal=True)  # (B,)
                all_errors.append(errors.cpu())
                all_kappas.append(kappa.squeeze(-1).cpu())

        all_errors = torch.cat(all_errors)  # (total,)
        all_kappas = torch.cat(all_kappas)

        mean_error = all_errors.mean().item()
        acc5 = (all_errors < 5.0).float().mean().item() * 100
        acc10 = (all_errors < 10.0).float().mean().item() * 100

        if verbose:
            median_error = all_errors.median().item()
            mean_kappa = all_kappas.mean().item()
            print("  Median Error: %.2f deg" % median_error)
            print("  Mean kappa: %.2f" % mean_kappa)
            # Kappa-error correlation (uncertainty calibration)
            correlation = torch.corrcoef(torch.stack([all_kappas, -all_errors]))[
                0, 1
            ].item()
            print("  Kappa-accuracy correlation: %.3f" % correlation)

        return mean_error, acc5, acc10
