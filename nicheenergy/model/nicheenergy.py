"""AnnData interface for intrinsic and niche-state representation learning."""

import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import Delaunay
import pytorch_lightning as pl
import torch.optim as optim
from scvi import settings
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import issparse
from scipy.stats import f
from statsmodels.stats.multitest import fdrcorrection


from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp
from scvi.train import TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split


from nicheenergy.module.nicheenergy import NicheEnergyModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class NicheEnergy(BaseModelClass):
    """
        Learn intrinsic and niche states within a variational autoencoder.

        Disentangled representation learning and state-modulated Dirichlet energy
        support label-free spatial niche identification.

        Parameters
        ----------
        adata : AnnData
                AnnData object that has been registered via `NicheEnergy.setup_anndata`.
        n_batch : int, default: 0
                Number of batches.
        n_hidden : int, default: 128
                Number of nodes per hidden layer.
        n_intrinsic : int, default: 20
                Dimensionality of the intrinsic state z.
        n_spatial : int, default: 20 
                Dimensionality of the niche state s.
        n_layers : int, default: 1
                Number of decoder layers. 
        dropout_rate : float, default: 0
                Dropout rate for neural networks.
        use_observed_lib_size : bool, default: True
                Use observed library size for RNA as scaling factor in mean of conditional distribution.
        lam_mi : float, default: 3000
                Coefficient of the independence regularization term. When using the mmd option, 
                the default coefficient is 3000. When using the mi option,
                the value of 5 is recommended.
        reg_to_use : str, default: 'mmd'
                The regularization method to use. Either 'mmd' (Maximal Mean Discrepancy) or 'mi' (Closed-form mutual information).
        dis_to_use : str, default: 'zinb'
                Observation model used for reconstruction loss.
                Supported options are: 'zinb', 'nb', 'poisson', 'normal',
                'normal_feature_scale', 'student_t', and 'mixture_nb'. 
                'normal_feature_scale' and 'student_t' are continuous
                likelihoods with a learned or fixed scale for each feature.
        continuous_scale_init : float, default: 1.0
                Initial per-feature scale for ``normal_feature_scale`` and
                ``student_t``. A value of 1 is appropriate for standardized
                continuous protein intensities.
        student_t_df : float, default: 4.0
                Degrees of freedom for the Student-t observation model.
        continuous_scale_mode : str, default: 'learned'
                Whether the per-feature continuous scale is optimized
                (``'learned'``) or held at ``continuous_scale_init``
                (``'fixed'``). Only used by ``normal_feature_scale`` and
                ``student_t``. The learned default preserves earlier behavior.
        var_eps : float, default: 1e-4
                Minimal variance for the variational posteriors.
        kl_weight : float, default: 0.5
                The KL divergence coefficient for the intrinsic state z.
        kl_s_weight : float, default: 0.01
                The KL divergence coefficient for the niche state s.
        attention_heads : int, default: 1
                The number of attention heads.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_intrinsic: int = 20,
        n_spatial: int = 20,
        n_layers: int = 1,
        dropout_rate: float = 0,
        use_observed_lib_size: bool = True,
        lam_mi: float = 3000,
        reg_to_use: str = 'mmd',
        dis_to_use: str = 'zinb',
        var_eps: float = 1e-4,
        kl_weight: float = 0.5,
        kl_s_weight: float = 0.01,
        attention_heads: int = 1,
        spatial_encoder_type: str = "xattn",
        distance_aware: bool = True,
        distance_emb_dim: int = 16,
        return_attention: bool = False,
        var_mode: str = "mlp",
        dirichlet_weight: float = 3.0,
        dirichlet_on: str = "mean",
        dirichlet_mode: str = "intrinsic_aware",
        dirichlet_dist_alpha: float = 1.0,
        dirichlet_dist_beta: float = 0.3,
        smoothness_weight: Optional[float] = None,
        smoothness_on: Optional[str] = None,
        smoothness_mode: Optional[str] = None,
        smoothness_dist_alpha: Optional[float] = None,
        smoothness_dist_beta: Optional[float] = None,
        smoothness_sim_kernel: Optional[str] = None,
        smoothness_detach_z: Optional[bool] = None,
        use_z_adversarial: bool = True,
        adv_hidden_dim: int = 32,
        adv_dropout: float = 0.1,
        adv_on: str = "sample",
        continuous_scale_init: float = 1.0,
        student_t_df: float = 4.0,
        continuous_scale_mode: str = "learned",
    ) -> None:
        super(NicheEnergy, self).__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = NicheEnergyModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_output=n_intrinsic,
            n_spatial=n_spatial,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            lam_mi=lam_mi,
            reg_to_use=reg_to_use,
            dis_to_use=dis_to_use,
            continuous_scale_mode=continuous_scale_mode,
            continuous_scale_init=continuous_scale_init,
            student_t_df=student_t_df,
            var_eps=var_eps,
            kl_weight=kl_weight,
            kl_s_weight=kl_s_weight,
            heads=attention_heads,
            spatial_encoder_type=spatial_encoder_type,
            distance_aware=distance_aware,
            distance_emb_dim=distance_emb_dim,
            return_attention=return_attention,
            var_mode=var_mode,
            dirichlet_weight=dirichlet_weight,
            dirichlet_on=dirichlet_on,
            dirichlet_mode=dirichlet_mode,
            dirichlet_dist_alpha=dirichlet_dist_alpha,
            dirichlet_dist_beta=dirichlet_dist_beta,
            smoothness_weight=smoothness_weight,
            smoothness_on=smoothness_on,
            smoothness_mode=smoothness_mode,
            smoothness_dist_alpha=smoothness_dist_alpha,
            smoothness_dist_beta=smoothness_dist_beta,
            smoothness_sim_kernel=smoothness_sim_kernel,
            smoothness_detach_z=smoothness_detach_z,
            use_z_adversarial=use_z_adversarial,
            adv_hidden_dim=adv_hidden_dim,
            adv_dropout=adv_dropout,
            adv_on=adv_on,
        )
        self._model_summary_string = "NicheEnergy"
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    #--------------------------------------------------
    # Preprocessing: setup_anndata, extract_edge_index
    #--------------------------------------------------

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        is_count_data: bool = True,
        **kwargs,
    ):
        """
        Set up AnnData instance for NicheEnergy model. A standard function to call in scvi-tools pipeline.

        Parameters
        ----------
        adata : AnnData
                AnnData object containing raw counts. Rows represent cells, columns represent features.
        layer : str, optional
                If not None, uses this as the key in adata.layers for raw count data.
        batch_key : str, optional
                Key in `adata.obs` for batch information. Categories will automatically be converted 
                into integer categories and saved to `adata.obs["_scvi_batch"]`. If None, assign 
                the same batch to all the data.
        labels_key : str, optional
                Key in `adata.obs` for label information. Categories will automatically be converted
                into integer categories and saved to `adata.obs["_scvi_labels"]`. If None, assign
                the same label to all the data.
        size_factor_key : str, optional
                Key in `adata.obs` for size factor information. 
        categorical_covariate_keys : List[str], optional
                Keys in `adata.obs` corresponding to categorical data. Not used in NicheEnergy.
        continuous_covariate_keys : List[str], optional
                Keys in `adata.obs` corresponding to continuous data. Not used in NicheEnergy.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def extract_edge_index(
        adata: AnnData,
        batch_key: Optional[str] = None,
        spatial_key: Optional[str] = 'spatial',
        method: str = 'knn',
        n_neighbors: int = 40,
        knn_edge_orientation: str = 'legacy_focal_to_neighbor',
        ):
        """
        Define edge_index for NicheEnergy model training.

        Parameters
        ----------
        adata : AnnData
                AnnData object.
        batch_key : str, optional
                Key in `adata.obs` for batch information. If batch_key is none,
                assume the adata is from the same batch. Otherwise, we create edge_index
                based on each batch and concatenate them.
        spatial_key : str, optional, default: 'spatial'
                Key in `adata.obsm` for spatial location.
        method : str, default: 'knn'
                Method for establishing the graph proximity relationship between
                cells. Two available methods are: knn and Delouney. Knn is used as default
                due to its flexible neighbor number selection.
        n_neighbors : int, default: 40
                The number of n_neighbors of knn graph. Not used if the graph
                is based on Delouney triangularization.
        knn_edge_orientation : str, default: 'legacy_focal_to_neighbor'
                Direction used when converting the scikit-learn KNN adjacency
                matrix to PyG ``edge_index``. ``'neighbor_to_focal'`` exports
                ``[selected neighbour, focal cell]``, matching DACA's
                sender/receiver convention. ``'legacy_focal_to_neighbor'``
                preserves the historical untransposed conversion used by the
                frozen benchmark checkpoints. The legacy default is retained
                while the direction-corrected benchmark is evaluated.

        Returns
        -------
        edge_index : torch.Tensor
                The edge index tensor containing the graph structure.
        """
        valid_orientations = {
            'legacy_focal_to_neighbor',
            'neighbor_to_focal',
        }
        if knn_edge_orientation not in valid_orientations:
            raise ValueError(
                "knn_edge_orientation must be one of "
                f"{sorted(valid_orientations)}, got {knn_edge_orientation!r}."
            )

        def orient_knn_edges(edge_index):
            # sklearn.neighbors.kneighbors_graph stores A[focal, neighbour].
            # PyG/DACA consumes edge_index[0]=sender, edge_index[1]=receiver.
            # Flipping the two rows therefore makes every selected neighbour
            # send to the focal cell without changing edge-column order.
            if knn_edge_orientation == 'neighbor_to_focal':
                return edge_index.flip(0).contiguous()
            return edge_index

        if batch_key is not None:
            j = 0
            for i in adata.obs[batch_key].unique():
                adata_tmp = adata[adata.obs[batch_key]==i].copy()
                if method == 'knn':
                    A = kneighbors_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                    edge_index_tmp, edge_weight = from_scipy_sparse_matrix(A)
                    edge_index_tmp = orient_knn_edges(edge_index_tmp)
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)

                else:
                    tri = Delaunay(adata_tmp.obsm[spatial_key])
                    triangles = tri.simplices
                    edges = set()
                    for triangle in triangles:
                        for i in range(3):
                            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                            edges.add(edge)
                    edge_index_tmp = torch.tensor(list(edges)).t().contiguous()
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)
        else:
            if method == 'knn':
                A = kneighbors_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
                edge_index, edge_weight = from_scipy_sparse_matrix(A)
                edge_index = orient_knn_edges(edge_index)
            else:
                tri = Delaunay(adata.obsm[spatial_key])
                triangles = tri.simplices
                edges = set()
                for triangle in triangles:
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        edges.add(edge)
                edge_index = torch.tensor(list(edges)).t().contiguous().type(torch.LongTensor)

        return edge_index

    def extract_edge_index_and_distance(
        adata: AnnData,
        batch_key: Optional[str] = None,
        spatial_key: Optional[str] = 'spatial',
        method: str = 'knn',
        n_neighbors: int = 40,
        knn_edge_orientation: str = 'legacy_focal_to_neighbor',
    ):
        """
        Define edge_index and edge_distance for distance-aware spatial encoder.

        Returns
        -------
        edge_index : torch.Tensor
            Graph connectivity in PyG format.
        edge_distance : torch.Tensor
            Euclidean distance for each edge in edge_index.

        Notes
        -----
        Use ``knn_edge_orientation='neighbor_to_focal'`` for DACA-consistent
        KNN aggregation. The legacy default reproduces frozen checkpoints.
        """
        edge_index = NicheEnergy.extract_edge_index(
            adata=adata,
            batch_key=batch_key,
            spatial_key=spatial_key,
            method=method,
            n_neighbors=n_neighbors,
            knn_edge_orientation=knn_edge_orientation,
        )

        coords = torch.tensor(adata.obsm[spatial_key], dtype=torch.float32)
        sender = edge_index[0]
        receiver = edge_index[1]
        edge_distance = torch.norm(coords[sender] - coords[receiver], dim=1)

        return edge_index, edge_distance

    #--------------------------------------------------------------------------------------------
    # Post-Training / Inference: get_latent_representation, get_attention, get_archetypes, get_se
    #--------------------------------------------------------------------------------------------

    @torch.no_grad()
    def get_latent_representation(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.

        Parameters
        ----------
        adata : AnnData, optional
                AnnData object with equivalent structure to initial AnnData. If `None`,
                defaults to the AnnData object used to initialize the model.
        indices : Sequence[int], optional
                Indices of cells in adata to use. If `None`, all cells are used.
        give_mean : bool, default: True
                Give mean of distribution or sample from it.
        batch_size : int, optional
                Mini-batch size for data loading into model. Defaults to full batch training.
        representation_kind : str, default: "all"
                "intrinsic" returns intrinsic states z; "interaction" returns niche
                states s; "all" returns their concatenation.

        Returns
        -------
        np.ndarray
                A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["intrinsic", "interaction","output","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
        outputs = self.module.inference(data,edge_index,eval_mode=True)
        latent = []
        if representation_kind == "intrinsic":
            latent_m = outputs["q_m"]
            latent_sample = outputs["z"]
        elif representation_kind == "interaction":
            latent_m = outputs["qgat_m"]
            latent_sample = outputs["z_gat"]
        elif representation_kind == "output":
            latent_m = self.module.gat_mean.lin_r(outputs["q_m"][:,-self.module.n_spatial:])
            latent_sample = latent_m
        elif representation_kind == "all":
            latent_m = outputs["qall_m"]
            latent_sample = outputs["z_all"]

        if give_mean:
            latent_sample = latent_m

        latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()
    

    @torch.no_grad()
    def get_attention_dict(self, edge_index):
        """
        Return attention dictionary for distance-aware cross-attention model.
        Works for spatial_encoder_type='xattn'.

        Returns
        -------
        dict
            Contains:
            - edge_index
            - attn_scores
            - attn_weights
            - distance_bias
            - edge_distance
        """
        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data("X").shape[0])]
        device = next(self.module.base_encoder.parameters()).device

        for key, value in data.items():
            data[key] = torch.Tensor(value).to(device)

        outputs = self.module.inference(data, edge_index, eval_mode=True)
        attn_dict = outputs.get("attention_dict", None)

        if attn_dict is None:
            raise ValueError(
                "No attention_dict found. Make sure spatial_encoder_type='xattn' "
                "and return_attention=True when initializing the model."
            )

        out = {}
        for k, v in attn_dict.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu()
            else:
                out[k] = v
        return out
      
    @torch.no_grad()
    def get_gene_level_influence(
        self,
        edge_index,
        batch_size_receivers: int = 256,
    ):
        """
        Compute GITIII-like edge-level gene influence strength for NicheEnergy.

        For each directed edge j -> i:
            1. obtain full prediction x_hat_i = Decoder(z_i, s_i)
            2. remove the edge contribution delta s_ij from s_i
            3. recompute x_hat_i^(-j) = Decoder(z_i, s_i - delta s_ij)
            4. define edge-level gene influence:
                   I_ijg = x_hat_ig - x_hat_ig^(-j)
            5. return edge-level abs strength:
                   abs_strength_ij = mean_g |I_ijg|

        Parameters
        ----------
        edge_index
            Either edge_index tensor or (edge_index, edge_distance) tuple.
        batch_size_receivers : int
            Number of receiver cells processed at once.

        Returns
        -------
        dict
            {
                "edge_index": np.ndarray, shape (2, E)
                "edge_distance": np.ndarray, shape (E,)
                "abs_strength": np.ndarray, shape (E,)
            }
        """
        device = next(self.module.parameters()).device

        # ----------------------------------------
        # Prepare full dataset on device
        # ----------------------------------------
        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data("X").shape[0])]
        for key, value in data.items():
            data[key] = torch.as_tensor(value, device=device)

        # ----------------------------------------
        # Run one full inference pass
        # ----------------------------------------
        outputs = self.module.inference(data, edge_index, eval_mode=True)

        q_m = outputs["q_m"]               # (N, n_intrinsic)
        qgat_m = outputs["qgat_m"]         # (N, n_spatial)
        library = outputs["library"]       # (N, 1)
        batch_index = outputs["batch_index"]
        attention_dict = outputs["attention_dict"]

        if attention_dict is None:
            raise ValueError(
                "attention_dict is None. Make sure return_attention=True "
                "and spatial_encoder_type='xattn'."
            )

        edge_contrib_s = attention_dict["edge_contrib_s"]   # (E, n_spatial)

        # ----------------------------------------
        # Parse edge_index / edge_distance
        # ----------------------------------------
        if isinstance(edge_index, tuple):
            edge_index_tensor, edge_distance = edge_index
        else:
            edge_index_tensor = edge_index
            edge_distance = None

        edge_index_tensor = edge_index_tensor.to(device)
        if edge_distance is not None:
            edge_distance = edge_distance.to(device)

        receiver = edge_index_tensor[1]
        n_edges = edge_index_tensor.shape[1]

        # ----------------------------------------
        # Full prediction x_hat
        # ----------------------------------------
        full_z_all = torch.cat((qgat_m, q_m), dim=1)
        full_gen = self.module.generative(
            {
                "z": q_m,
                "z_all": full_z_all,
                "library": library,
                "batch_index": batch_index,
            }
        )
        full_px_rate = full_gen["px_rate"]   # (N, G)

        # ----------------------------------------
        # Group incoming edges by receiver
        # ----------------------------------------
        edge_order = torch.argsort(receiver)
        receiver_sorted = receiver[edge_order]

        unique_receivers, counts = torch.unique_consecutive(receiver_sorted, return_counts=True)
        offsets = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(counts, dim=0)
            ]
        )

        abs_strength = torch.zeros(n_edges, device=device, dtype=full_px_rate.dtype)

        # ----------------------------------------
        # Process receivers in chunks
        # ----------------------------------------
        num_unique = unique_receivers.shape[0]

        for start in range(0, num_unique, batch_size_receivers):
            end = min(start + batch_size_receivers, num_unique)

            edge_blocks = []
            receiver_ids = []

            for idx in range(start, end):
                r = unique_receivers[idx]
                s_idx = offsets[idx].item()
                e_idx = offsets[idx + 1].item()
                edge_ids = edge_order[s_idx:e_idx]   # global edge ids for this receiver

                edge_blocks.append(edge_ids)
                receiver_ids.append(r)

            if len(edge_blocks) == 0:
                continue

            # Flatten all edges in this receiver chunk
            edge_ids_chunk = torch.cat(edge_blocks, dim=0)          # (M,)
            receiver_ids_chunk = torch.cat(
                [
                    torch.full((len(edge_ids),), r, device=device, dtype=torch.long)
                    for edge_ids, r in zip(edge_blocks, receiver_ids)
                ],
                dim=0
            )                                                      # (M,)

            # Full states for corresponding receivers
            z_chunk = q_m[receiver_ids_chunk]                      # (M, n_intrinsic)
            s_chunk = qgat_m[receiver_ids_chunk]                   # (M, n_spatial)
            lib_chunk = library[receiver_ids_chunk]                # (M, 1)
            batch_chunk = batch_index[receiver_ids_chunk]          # (M, ...)

            # Remove one edge contribution at a time
            s_minus = s_chunk - edge_contrib_s[edge_ids_chunk]     # (M, n_spatial)

            z_all_minus = torch.cat((s_minus, z_chunk), dim=1)     # (M, n_spatial + n_intrinsic)

            gen_minus = self.module.generative(
                {
                    "z": z_chunk,
                    "z_all": z_all_minus,
                    "library": lib_chunk,
                    "batch_index": batch_chunk,
                }
            )
            px_minus = gen_minus["px_rate"]                        # (M, G)

            px_full_chunk = full_px_rate[receiver_ids_chunk]       # (M, G)

            delta = px_full_chunk - px_minus                       # (M, G)
            abs_strength_chunk = delta.abs().mean(dim=1)           # (M,)

            abs_strength[edge_ids_chunk] = abs_strength_chunk

        out = {
            "edge_index": edge_index_tensor.detach().cpu().numpy(),
            "abs_strength": abs_strength.detach().cpu().numpy(),
        }

        if edge_distance is not None:
            out["edge_distance"] = edge_distance.detach().cpu().numpy()
        else:
            out["edge_distance"] = None

        return out
    
    
    @torch.no_grad()
    def get_archetypes(
        self,
        embedding,
        noc=5,
        delta=0.1,
        conv_crit=0.00001,
        maxiter=200,
        verbose=False,
    ) -> np.ndarray:
        """
        Calculate archetypal analysis for the input latent representation. A preliminary step for `get_se` function.

        Parameters
        ----------
        embedding : np.ndarray
               Input data matrix where each row represents a cell.
        noc : int, default: 5
                Number of archetypes to extract.
        delta : float, default: 0.1
                The relaxation parameter in PCHA algorithm.
        conv_crit : float, default: 0.00001
                Convergence criterion. Algorithm stops when the relative change in objective function is less than this.
        maxiter : int, default: 200
                Maximum number of iterations.
        verbose : bool, default: False
                Whether to print progress during optimization.
 
        Returns
        -------
        Tuple
                Returns a tuple containing:
                - Feature loading matrix (shape: (n_archetypes, latent_dim))
                - Archetypal representation matrix (shape: (n_cells, n_archetypes))
                - Explained variance ratio
        """

        from py_pcha import PCHA
        XC, S, C, SSE, varexpl = PCHA(embedding.T, noc=noc, delta=delta,conv_crit=conv_crit, maxiter=maxiter, verbose=verbose)
        
        return XC.T, S, varexpl    
    
    @torch.no_grad()    
    def get_se(
        self,
        edge_index: Optional[torch.Tensor] = None,
        adata: Optional[AnnData] = None,
        z_label: Optional[str] = 'nicheenergy_z',
        s_label: Optional[str] = 'nicheenergy_s',
        transformation = 'log1p',
        batch_label = None,
        num_arch = 5,
        delta = 0.1,
        maxiter = 200,
        Kfold = 5,
        eps = 0,
        thres = 0.95,
        positivity_filter = False,
        cell_type_label = None,
        obsm_label = None,
        mode = 'individual',
    ) -> np.ndarray:
        """
        Return the spatial effect for each cell in spatial omics data. Requires training the NicheEnergy model in priori.  

        Parameters
        ----------
        edge_index : torch.Tensor
                The object created by function "extract_edge_index".
        adata : AnnData, optional
                AnnData object with equivalent structure to initial AnnData. If `None`,
                defaults to the AnnData object used to initialize the model.
        z_label : str, optional
                The key for intrinsic states z in adata.obsm. If adata is `None`, 
                then it is calculated in this function.
        s_label : str, optional
                The key for niche states s in adata.obsm. If adata is `None`, 
                then it is calculated in this function.
        transformation : str, default: 'log1p'
                If `log1p`, perform log1p on a copy of the data. Else, operate on the given adata.X.
        batch_label : str, optional
                If given, then add it as a covariate in the double machine learning model.
        num_arch : int, default: 5
                Number of archetypes in archetypal transformation.
        delta : float, default: 0.1
                Delta parameter in archetypal transformation.
        maxiter : int, default: 200
                Maximum iterations in archetypal transformation.
        Kfold : int, default: 5
                Number of folds in cross validation.
        eps : float, default: 0
                Epsilon parameter in archetypal transformation.
        thres : float, default: 0.95
                Thres2 in positivity index calculation.
        positivity_filter : bool, default: False
                If True, only return the spatial effect of cells satisfying positivity condition,
                and return the indices of these cells.
        cell_type_label : str, optional
                If given, then add it as a covariate in the double machine learning model.
        obsm_label : str, optional
                If given, then add it as a covariate in the double machine learning model. 

        Returns
        -------
        Union[tuple, tuple]
                If positivity is `False`, return (spatial_effect, R2s, p_values, archetypes).
                If positivity is `True`, return (positive_indices, spatial_effect, R2s, p_values, archetypes). 
        """
        
        if adata is None:
            data = AnnTorchDataset(self.adata_manager)
            data = data[np.arange(data.get_data('X').shape[0])]
            for key, value in data.items():
                data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
            outputs = self.module.inference(data,edge_index,eval_mode=True)
        
            batch_index = outputs["batch_index"]
            latent_z = outputs["q_m"].detach().cpu().numpy()
            latent_s = outputs["qgat_m"].detach().cpu().numpy()
        
            adata_tmp = self.adata_manager.adata.copy()
            
        else:
            adata_tmp = adata.copy()
            
            latent_z = adata_tmp.obsm[z_label]
            latent_s = adata_tmp.obsm[s_label]
            
        ## estimate generalized propensity score
        if transformation == 'log1p':
            sc.pp.normalize_total(adata_tmp)
            sc.pp.log1p(adata_tmp)
            
        if batch_label is not None:
            df = pd.get_dummies(adata_tmp.obs[batch_label]).values
            latent_z = np.hstack((latent_z,df))
            
        if cell_type_label is not None:
            df2 = pd.get_dummies(adata_tmp.obs[cell_type_label]).values
            latent_z = np.hstack((latent_z,df2))
            #df2_ = df2 / df2.sum(axis=0) 
            
        if obsm_label is not None:
            df3 = np.asarray(adata_tmp.obsm[obsm_label])
            latent_z = np.hstack((latent_z,df3))

        if issparse(adata_tmp.X):
            adata_tmp.X = adata_tmp.X.toarray()
            
        arc, S, varexpl = self.get_archetypes(latent_s,noc=num_arch,delta=delta,maxiter=maxiter)
        
        ## S is the continuous treatment variable
        S = np.asarray(S.T)
        
        if positivity_filter:
            sc.pp.neighbors(adata_tmp,use_rep=z_label)
            sc.tl.leiden(adata_tmp,resolution=0.6)
            df = pd.DataFrame(S.copy())
            df = (df>0.5).astype(int)
            df['cluster'] = adata_tmp.obs['leiden'].values.copy()
            df = df.loc[df.sum(axis=1)>0]
            df_ = pd.DataFrame(df.groupby('cluster').mean().max(axis=0))
            
            positive_indices = ((pd.DataFrame(S.copy(),index=adata_tmp.obs_names).loc[:,df_.values>=thres]>0.5).sum(axis=1) == 0)
            adata_tmp = adata_tmp[positive_indices]
            latent_z = latent_z[positive_indices]
            latent_s = latent_s[positive_indices]
            S = S[positive_indices]
            S = S[:,df_.values.flatten()<thres]
        
        np.random.seed(42)
        indices = np.arange(latent_z.shape[0])
        np.random.shuffle(indices)
        split_data = np.array_split(indices, Kfold)        
        
        
        
        if mode == 'individual':     
            se_list = []
            r2_zlist = []
            r2_slist = []
            r2_zpvlist = []
            r2_spvlist = []
            for i in range(S.shape[1]):
                se = np.zeros((latent_z.shape[0],adata_tmp.shape[1]))
                r2_z = np.zeros(adata_tmp.shape[1])
                r2_s = np.zeros(adata_tmp.shape[1])
                for ind in split_data:
                    Si = S[ind,i]
                    lr = LinearRegression()
                    lr.fit(latent_z[ind],Si)
                    lr2 = LinearRegression()
                    lr2.fit(latent_z[ind],adata_tmp.X[ind])
                    Si_ = Si - lr.predict(latent_z[ind])
                    X_ = adata_tmp.X[ind] - lr2.predict(latent_z[ind])
                    lr3 = LinearRegression()
                    lr3.fit(latent_z[ind],X_ / (Si_[:,None]+eps),sample_weight = Si_ ** 2)
                    se[ind] = lr3.predict(latent_z[ind]) * Si_[:,None]
                    r2_z = r2_z + r2_score(adata_tmp.X[ind],lr2.predict(latent_z[ind]), multioutput='raw_values')
                    r2_s = r2_s + r2_score(X_, lr3.predict(latent_z[ind]) * Si_[:,None], multioutput='raw_values')
                
            #if cell_type_label is not None:
                
            #    ct_means = pd.DataFrame(se)
            #    ct_means[cell_type_label] = adata_tmp.obs[cell_type_label].values.copy()
            #    ct_means = ct_means.groupby(cell_type_label).median()
            #    #pd.DataFrame(df2_.T @ se, index = pd.get_dummies(adata_tmp.obs[cell_type_label]).columns.copy())
            #    se = se - ct_means.loc[adata_tmp.obs[cell_type_label].values].values
                
                
                se_list.append(se)
                r2_zlist.append(r2_z/Kfold)
                r2_slist.append(r2_s/Kfold)
                r2_zpvlist.append(return_f_pv(adata_tmp.X[ind],r2_z/Kfold))
                r2_spvlist.append(return_f_pv(latent_z[ind],r2_s/Kfold))
            if positivity_filter:
                return positive_indices, se_list, r2_zlist, r2_slist, r2_zpvlist, r2_spvlist, S
            else:
                return se_list, r2_zlist, r2_slist, r2_zpvlist, r2_spvlist, S
        else:
            se = np.zeros((latent_z.shape[0],adata_tmp.shape[1]))
            r2_z = np.zeros(adata_tmp.shape[1])
            r2_s = np.zeros(adata_tmp.shape[1])
            for ind in split_data:
                lr = LinearRegression()
                lr.fit(latent_z[ind],S[ind])
                lr2 = LinearRegression()
                lr2.fit(latent_z[ind],adata_tmp.X[ind])
                S_ = S[ind] - lr.predict(latent_z[ind])
                X_ = adata_tmp.X[ind] - lr2.predict(latent_z[ind])
                lr3 = LinearRegression()
                design_matrix = (latent_z[ind][:,:,None] * S_[:,None,:]).reshape(latent_z[ind].shape[0],latent_z.shape[1] * S_.shape[1])
                design_matrix = np.hstack((design_matrix,S_))
                lr3.fit(design_matrix,X_)
                se[ind] = lr3.predict(design_matrix)
                r2_z = r2_z + r2_score(adata_tmp.X[ind],lr2.predict(latent_z[ind]), multioutput='raw_values')
                r2_s = r2_s + r2_score(X_, lr3.predict(design_matrix), multioutput='raw_values')
            if positivity_filter:
                return positive_indices, se, r2_z/Kfold, r2_s/Kfold, return_f_pv(adata_tmp.X[ind],r2_z/Kfold), return_f_pv(adata_tmp.X[ind],r2_s/Kfold), S
            else:
                return se, r2_z/Kfold, r2_s/Kfold, return_f_pv(adata_tmp.X[ind],r2_z/Kfold), return_f_pv(adata_tmp.X[ind],r2_s/Kfold), S
                
    #------------------------
    # Main Training Function
    #------------------------

    def train(
        self,
        edge_index: torch.Tensor,
        max_epochs: Optional[int] = 100,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        batch_size: Optional[int] = 512,
        anneal_epochs: int = 50,
        mae_epochs: int = 0,
        validation_size: Optional[float] = None,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
        adv_weight: float = 10.0,
        adv_lr: float = 1e-3,
        adversarial_step: int = 1,
    ) -> None:
        """
        Fit the model on expression and spatial graph inputs; return training and validation losses.
        """
        if max_epochs is None:
            n_cells = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        n_cells = self.adata_manager.adata.n_obs

        if validation_size is None:
            if n_cells <= 15000:
                validation_size = 1 - train_size
            else:
                validation_size = min(1500 / n_cells, 1 - train_size)

        n_train, n_val = validate_data_split(
            self.adata_manager.adata.n_obs,
            train_size,
            validation_size,
        )

        random_state = np.random.RandomState(seed=settings.seed)
        indices = np.arange(self.adata_manager.adata.n_obs)
        random_state.shuffle(indices)
        train_mask = indices[:n_train]
        val_mask = indices[n_train:(n_val + n_train)]
        test_mask = indices[(n_val + n_train):]

        if device is None:
            if use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.module = self.module.to(device)

        if isinstance(edge_index, tuple):
            edge_index = tuple(e.to(device) for e in edge_index)
        else:
            edge_index = edge_index.to(device)

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(device)

        use_adv = bool(
            getattr(self.module, "use_z_adversarial", False)
            and adv_weight > 0
            and getattr(self.module, "classifier_z", None) is not None
        )

        if use_adv:
            classifier_params = list(self.module.classifier_z.parameters())
            classifier_param_ids = set(id(p) for p in classifier_params)

            main_params = [
                p for p in self.module.parameters()
                if id(p) not in classifier_param_ids
            ]

            optimizer = optim.Adam(
                main_params,
                lr=lr,
                weight_decay=weight_decay,
            )
            optimizer_discriminator = optim.Adam(
                classifier_params,
                lr=adv_lr,
                weight_decay=weight_decay,
            )

            print(
                f"Using z-only adversarial batch loss: "
                f"adv_weight={adv_weight}, adv_lr={adv_lr}, "
                f"adversarial_step={adversarial_step}, adv_on={self.module.adv_on}",
                flush=True,
            )
        else:
            optimizer = optim.Adam(
                self.module.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            optimizer_discriminator = None

            if adv_weight > 0:
                print(
                    "adv_weight > 0 was provided, but z adversarial is disabled. "
                    "Check use_z_adversarial=True and n_batch > 1.",
                    flush=True,
                )
        train_loss = []
        val_loss = []
        pbar = tqdm(range(1, max_epochs + 1))

        if batch_size is not None:
            batch_indices = [
                train_mask[i:i + batch_size]
                for i in range(0, train_mask.shape[0], batch_size)
            ]
            train_loader = {}
            for i, batch_index in enumerate(batch_indices):
                data_masked = {}
                for key, value in data.items():
                    if value is None:
                        data_masked[key] = None
                    else:
                        data_masked[key] = value[batch_index]
                train_loader[i] = data_masked
        else:
            data_masked = {}
            for key, value in data.items():
                if value is None:
                    data_masked[key] = None
                else:
                    data_masked[key] = value[train_mask]
            train_loader = data_masked

        val_loader = {}
        for key, value in data.items():
            if value is None:
                val_loader[key] = None
            else:
                val_loader[key] = value[val_mask]

        for epoch in pbar:
            weight = min(1.0, epoch / anneal_epochs)
            if epoch < mae_epochs:
                eval_mode = False
            else:
                eval_mode = True

            train_loss.append(
                _train(
                    self.module,
                    data,
                    edge_index,
                    train_mask,
                    train_loader,
                    optimizer,
                    batch_size,
                    weight,
                    eval_mode,
                    optimizer_discriminator=optimizer_discriminator,
                    adv_weight=adv_weight if use_adv else 0.0,
                    adversarial_step=adversarial_step,
                )
            )
            val_loss.append(
                _eval(
                    self.module,
                    data,
                    edge_index,
                    val_mask,
                    val_loader,
                    weight,
                )
            )

            pbar.set_description(f"Epoch {epoch}/{max_epochs}")
            pbar.set_postfix(
                train_loss=train_loss[epoch - 1],
                val_loss=val_loss[epoch - 1].numpy(),
            )

        return train_loss, val_loss

#--------------------------------------------
# Helper Methods: _train, _eval, return_f_pv
#--------------------------------------------

def _subset_edge_index(edge_index, node_idx):
    """
    Subset a global edge_index (or (edge_index, edge_distance)) to the given node_idx,
    and relabel node ids to local batch indices [0, ..., len(node_idx)-1].
    """
    if isinstance(edge_index, tuple):
        edge_index_tensor, edge_distance = edge_index
    else:
        edge_index_tensor = edge_index
        edge_distance = None

    device = edge_index_tensor.device
    node_idx = torch.as_tensor(node_idx, device=device, dtype=torch.long)

    # global -> local mapping
    max_index = int(max(edge_index_tensor.max().item(), node_idx.max().item())) + 1
    mapping = torch.full((max_index,), -1, device=device, dtype=torch.long)
    mapping[node_idx] = torch.arange(node_idx.shape[0], device=device)

    src_global = edge_index_tensor[0]
    dst_global = edge_index_tensor[1]

    edge_mask = (mapping[src_global] >= 0) & (mapping[dst_global] >= 0)

    src_local = mapping[src_global[edge_mask]]
    dst_local = mapping[dst_global[edge_mask]]
    sub_edge_index = torch.stack([src_local, dst_local], dim=0)

    if edge_distance is not None:
        sub_edge_distance = edge_distance[edge_mask]
        return sub_edge_index, sub_edge_distance
    else:
        return sub_edge_index
      
      
def _train(
    model,
    data,
    edge_index,
    mask,
    train_loader,
    optimizer,
    batch_size,
    weight,
    eval_mode,
    optimizer_discriminator=None,
    adv_weight: float = 0.0,
    adversarial_step: int = 1,
):
    """
    Helper function for training. Has full-batch and mini-batch modes.

    If z adversarial is enabled:
      1. update discriminator to predict batch from sampled z;
      2. update main model with negative classifier loss to fool discriminator.
    """
    train_loss = []
    model.train()

    use_adv = bool(
        getattr(model, "use_z_adversarial", False)
        and adv_weight > 0
        and optimizer_discriminator is not None
        and getattr(model, "classifier_z", None) is not None
    )
    
    def _set_classifier_requires_grad(flag: bool):
        if use_adv:
            for p in model.classifier_z.parameters():
                p.requires_grad_(flag)
    
    def _update_discriminator(latent_dict_masked, data_masked):
        """
        Train z batch classifier only.
        Encoder outputs are detached, so only classifier_z is updated.
        """
        if not use_adv:
            return None

        _set_classifier_requires_grad(True)

        loss_disc_value = None
        for _ in range(adversarial_step):
            z_for_adv = model.get_z_for_adversarial(latent_dict_masked).detach()
            batch_index = data_masked[REGISTRY_KEYS.BATCH_KEY]

            loss_disc = model.classifier_loss_z(z_for_adv, batch_index) * adv_weight

            optimizer_discriminator.zero_grad()
            loss_disc.backward()
            optimizer_discriminator.step()

            loss_disc_value = loss_disc.detach()

        return loss_disc_value

    if batch_size is None:
        latent_dict = model.inference(data, edge_index, eval_mode=eval_mode)
        
        latent_dict_masked = {}
        for key, value in latent_dict.items():
            if value is None:
                latent_dict_masked[key] = None
            elif torch.is_tensor(value):
                latent_dict_masked[key] = value[mask]
            else:
                latent_dict_masked[key] = value

        # --------------------------------------------------
        # 1. Train discriminator on detached z
        # --------------------------------------------------
        if use_adv:
            _update_discriminator(latent_dict_masked, train_loader)

        optimizer.zero_grad()
        # --------------------------------------------------
        # 2. Train main model; adversarial term has negative sign inside model.loss
        # --------------------------------------------------
        decoder_dict = model.generative(latent_dict_masked)
        train_edge_index = _subset_edge_index(edge_index, mask)

        if use_adv:
            _set_classifier_requires_grad(False)

        lossrecorder = model.loss(
            train_loader,
            latent_dict_masked,
            decoder_dict,
            weight,
            edge_index=train_edge_index,
            adversarial_weight=adv_weight if use_adv else 0.0,
        )

        loss = lossrecorder.loss
        loss.backward()
        optimizer.step()

        if use_adv:
            _set_classifier_requires_grad(True)
            optimizer_discriminator.zero_grad(set_to_none=True)

        train_loss.append(loss.detach().cpu())

    else:
        batch_indices = [
            mask[i:i + batch_size]
            for i in range(0, mask.shape[0], batch_size)
        ]

        for i, batch_index in enumerate(batch_indices):
            latent_dict = model.inference(data, edge_index, eval_mode=eval_mode)

            latent_dict_masked = {}
            for key, value in latent_dict.items():
                if value is None:
                    latent_dict_masked[key] = None
                elif torch.is_tensor(value):
                    latent_dict_masked[key] = value[batch_index]
                else:
                    latent_dict_masked[key] = value

            data_masked = train_loader[i]

            # --------------------------------------------------
            # 1. Train discriminator on detached z
            # --------------------------------------------------
            if use_adv:
                _update_discriminator(latent_dict_masked, data_masked)
                
            optimizer.zero_grad()
            # --------------------------------------------------
            # 2. Train main model; adversarial term has negative sign inside model.loss
            # --------------------------------------------------
            decoder_dict = model.generative(latent_dict_masked)
            batch_edge_index = _subset_edge_index(edge_index, batch_index)

            if use_adv:
                _set_classifier_requires_grad(False)

            lossrecorder = model.loss(
                data_masked,
                latent_dict_masked,
                decoder_dict,
                weight,
                edge_index=batch_edge_index,
                adversarial_weight=adv_weight if use_adv else 0.0,
            )

            loss = lossrecorder.loss
            loss.backward()
            optimizer.step()

            if use_adv:
                _set_classifier_requires_grad(True)
                optimizer_discriminator.zero_grad(set_to_none=True)

            train_loss.append(loss.detach().cpu())

    return np.array(train_loss).mean()

def _eval(model, data, edge_index, mask, val_loader, weight):
    """
    Helper function for evaluation. Has full-batch and mini-batch modes.
    """
    model.eval()

    with torch.no_grad():
        latent_dict = model.inference(data, edge_index, eval_mode=True)

        latent_dict_masked = {}
        for key, value in latent_dict.items():
            if value is None:
                latent_dict_masked[key] = None
            elif torch.is_tensor(value):
                latent_dict_masked[key] = value[mask]
            else:
                latent_dict_masked[key] = value

        decoder_dict = model.generative(latent_dict_masked)
        val_edge_index = _subset_edge_index(edge_index, mask)

        lossrecorder = model.loss(
            val_loader,
            latent_dict_masked,
            decoder_dict,
            weight,
            edge_index=val_edge_index,
        )

    return lossrecorder.loss.detach().cpu()

def return_f_pv(X,Rsq):
    """
    Helper function for calculating p values of spatial effects.
    """
    N = X.shape[0]
    K = X.shape[1] + 1
    fstat = (Rsq/(1-Rsq))*((N-K-1)/K)

    df_model = X.shape[1]
    df_residuals = X.shape[0] - X.shape[1] + 1
    p_values = f.sf(fstat, df_model, df_residuals)
    
    rej, adj_p = fdrcorrection(p_values)
    return adj_p
        
