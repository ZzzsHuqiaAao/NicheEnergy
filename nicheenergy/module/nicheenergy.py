"""Neural modules for intrinsic and niche-state representation learning."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal, Poisson, StudentT
from torch.distributions import kl_divergence as kl
from torch.autograd import Variable

import torch.optim as optim

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import softmax as pyg_softmax

torch.backends.cudnn.benchmark = True

CONTINUOUS_LIKELIHOODS = {
    "normal",
    "normal_feature_scale",
    "student_t",
}

SUPPORTED_LIKELIHOODS = {
    "zinb",
    "nb",
    "poisson",
    *CONTINUOUS_LIKELIHOODS,
    "mixture_nb",
    "mnb",
}

class DistanceAwareSpatialCrossAttention(nn.Module):
    """
    Receiver-conditioned, distance-aware spatial cross-attention.

    Input
    -----
    x : Tensor, shape (N, d)
        Node features used for spatial interaction encoding.
    edge_index : LongTensor, shape (2, E)
        Graph edges in PyG format. We assume edge_index[0] = sender, edge_index[1] = receiver.
    edge_distance : Tensor, shape (E,), optional
        Pairwise distance for each edge.

    Output
    ------
    out : Tensor, shape (N, d)
        Aggregated spatial representation per receiver node.
    attn_dict : dict
        Optional dictionary with pairwise attention quantities.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 1,
        distance_aware: bool = True,
        distance_emb_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.distance_aware = distance_aware
        self.distance_emb_dim = distance_emb_dim
        self.dropout = dropout

        if out_channels % n_heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by n_heads ({n_heads})."
            )
        self.head_dim = out_channels // n_heads

        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)

        if self.distance_aware:
            self.distance_mlp = nn.Sequential(
                nn.Linear(5, distance_emb_dim),
                nn.ReLU(),
                nn.Linear(distance_emb_dim, n_heads),
            )
        else:
            self.distance_mlp = None

        self.out_proj = nn.Linear(out_channels, out_channels, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    @staticmethod
    def build_distance_features(edge_distance: torch.Tensor) -> torch.Tensor:
        """
        Build 5 distance features:
            1 / (1 + d)
            1 / (1 + d^2)
            1 / (1 + sqrt(d))
            exp(-d)
            exp(-d^2)
        """
        d = edge_distance.float().clamp_min(0.0)
        sqrt_d = torch.sqrt(d + 1e-12)

        feats = torch.stack(
            [
                1.0 / (1.0 + d),
                1.0 / (1.0 + d.pow(2)),
                1.0 / (1.0 + sqrt_d),
                torch.exp(-d),
                torch.exp(-d.pow(2)),
            ],
            dim=-1,
        )
        return feats

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distance: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        apply_dropout: Optional[bool] = None,
    ):
        n_nodes = x.size(0)
        sender = edge_index[0]
        receiver = edge_index[1]

        Q = self.W_q(x).view(n_nodes, self.n_heads, self.head_dim)
        K = self.W_k(x).view(n_nodes, self.n_heads, self.head_dim)
        V = self.W_v(x).view(n_nodes, self.n_heads, self.head_dim)

        q_i = Q[receiver]   # (E, H, D)
        k_j = K[sender]     # (E, H, D)
        v_j = V[sender]     # (E, H, D)

        # scaled dot-product score per edge and per head
        scores = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)  # (E, H)

        distance_bias = None
        if self.distance_aware and edge_distance is not None:
            dist_feat = self.build_distance_features(edge_distance)  # (E, 5)
            distance_bias = self.distance_mlp(dist_feat)            # (E, H)
            scores = scores + distance_bias
        else:
            distance_bias = torch.zeros_like(scores)

        # softmax over incoming edges of each receiver
        alpha = pyg_softmax(scores, receiver)  # (E, H)
        # apply_dropout controls the attention dropout explicitly, with no global side effect:
        #   None  -> honour the module's train/eval state (default behaviour, nn.Dropout is a no-op in eval);
        #   False -> force deterministic (skip attention dropout even in train mode) -- used by the
        #            Dirichlet energy branch so the only difference between compared arms is the gradient
        #            block, never a different dropout mask;
        #   True  -> force dropout on.
        if apply_dropout is not False:
            alpha = self.attn_dropout(alpha)

        # weighted aggregation to receiver nodes
        weighted_v = v_j * alpha.unsqueeze(-1)  # (E, H, D)

        out = torch.zeros(
            (n_nodes, self.n_heads, self.head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        out.index_add_(0, receiver, weighted_v)
        out = out.reshape(n_nodes, self.out_channels)
        out = self.out_proj(out)

        attn_dict = None
        if return_attention:
            # --------------------------------------------------
            # Per-edge contribution before and after out_proj
            # --------------------------------------------------
            # edge_contrib_preproj: contribution of each edge before out_proj
            # shape: (E, out_channels)
            edge_contrib_preproj = weighted_v.reshape(
                weighted_v.shape[0], self.out_channels
            )

            # edge_contrib_s: contribution of each edge to final spatial latent mean
            # IMPORTANT: do NOT include bias here, since bias is node-level and not edge-specific
            edge_contrib_s = F.linear(
                edge_contrib_preproj,
                self.out_proj.weight,
                bias=None,
            )  # (E, out_channels)

            attn_dict = {
                "edge_index": edge_index,
                "attn_scores": scores,
                "attn_weights": alpha,
                "distance_bias": distance_bias,
                "edge_distance": edge_distance,
                "edge_contrib_preproj": edge_contrib_preproj,
                "edge_contrib_s": edge_contrib_s,
            }

        return out, attn_dict



def mixture_nb_loss(x, beta, alpha, pi, theta, eps: float = 1e-8):
    """Two-component (background/foreground) mixture negative-binomial NLL.

    Adapted from the protein likelihood of scProca/scMuMA. The background
    component is ``NB(mu_bg=beta, theta)`` and the foreground component is
    ``NB(mu_fg=alpha*beta, theta)`` with ``alpha>1``; ``sigmoid(pi)`` is the
    background mixing weight. Returns the per-cell NLL summed over features.
    """
    mu_1 = beta
    mu_2 = beta * alpha
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    log_theta_mu_1 = torch.log(theta + mu_1 + eps)
    log_theta_mu_2 = torch.log(theta + mu_2 + eps)
    lgamma_x_theta = torch.lgamma(x + theta)
    lgamma_theta = torch.lgamma(theta)
    lgamma_x1 = torch.lgamma(x + 1)
    log_nb_1 = (
        theta * (torch.log(theta + eps) - log_theta_mu_1)
        + x * (torch.log(mu_1 + eps) - log_theta_mu_1)
        + lgamma_x_theta - lgamma_theta - lgamma_x1
    )
    log_nb_2 = (
        theta * (torch.log(theta + eps) - log_theta_mu_2)
        + x * (torch.log(mu_2 + eps) - log_theta_mu_2)
        + lgamma_x_theta - lgamma_theta - lgamma_x1
    )
    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi)), dim=0)
    log_mixture_nb = logsumexp - F.softplus(-pi)
    return -log_mixture_nb.sum(-1)


class NicheEnergyModule(nn.Module):
    def on_load(self, model):
        pass


    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_output: int = 20,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        reg_to_use: str = 'mmd',
        dis_to_use: str = 'zinb',
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        kl_weight: float = 1,
        kl_s_weight: float = 0.01,
        lam_mi: float = 3000,
        var_eps: float = 1e-4,
        heads = 1,
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
        dirichlet_sim_kernel: str = "half_cos",
        dirichlet_detach_z: bool = True,
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
    ):
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_spatial = n_spatial
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.reg_to_use = reg_to_use
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        if dis_to_use not in SUPPORTED_LIKELIHOODS:
            raise ValueError(
                f"Unknown dis_to_use: {dis_to_use}. "
                f"Choose from {sorted(SUPPORTED_LIKELIHOODS)}."
            )
        if (
            dis_to_use in {"normal_feature_scale", "student_t"}
            and continuous_scale_init <= 0
        ):
            raise ValueError("continuous_scale_init must be positive.")
        valid_scale_modes = {"learned", "fixed"}
        if continuous_scale_mode not in valid_scale_modes:
            raise ValueError(
                "continuous_scale_mode must be one of "
                f"{sorted(valid_scale_modes)}."
            )
        if (
            continuous_scale_mode == "fixed"
            and dis_to_use not in {"normal_feature_scale", "student_t"}
        ):
            raise ValueError(
                "continuous_scale_mode='fixed' is only valid for "
                "normal_feature_scale or student_t."
            )
        if dis_to_use == "student_t" and student_t_df <= 2:
            raise ValueError("student_t_df must be greater than 2.")

        self.dis_to_use = dis_to_use
        self.continuous_scale_mode = continuous_scale_mode
        self.continuous_scale_init = float(continuous_scale_init)
        self.student_t_df = float(student_t_df)
        if self.dis_to_use in {"normal_feature_scale", "student_t"}:
            initial_log_scale = torch.full(
                (n_input,),
                float(np.log(self.continuous_scale_init)),
            )
            if self.continuous_scale_mode == "learned":
                self.continuous_log_scale = nn.Parameter(initial_log_scale)
            else:
                # Keep the same state-dict key as learned-scale checkpoints,
                # while excluding the fixed value from optimizer parameters.
                self.register_buffer("continuous_log_scale", initial_log_scale)
        else:
            self.register_parameter("continuous_log_scale", None)
        self.use_observed_lib_size = use_observed_lib_size
        self.var_eps = var_eps
        self.lam_mi = lam_mi
        self.kl_weight = kl_weight
        # KL weight on the niche state s, produced by the Transformer cross-attention
        # encoder. (Internal qgat_*/zgat/kl_divergence_zgat names below are legacy and refer to
        # this same spatial-s latent, not to a graph-attention network.)
        self.kl_s_weight = kl_s_weight
        #self.weight_ = nn.Parameter(torch.ones(n_output)*0.5,requires_grad=True)
        self.spatial_encoder_type = spatial_encoder_type
        self.distance_aware = distance_aware
        self.distance_emb_dim = distance_emb_dim
        self.return_attention = return_attention
        self.var_mode = var_mode
        if smoothness_weight is not None:
            dirichlet_weight = smoothness_weight
        if smoothness_on is not None:
            dirichlet_on = smoothness_on
        if smoothness_mode is not None:
            dirichlet_mode = smoothness_mode
        if smoothness_dist_alpha is not None:
            dirichlet_dist_alpha = smoothness_dist_alpha
        if smoothness_dist_beta is not None:
            dirichlet_dist_beta = smoothness_dist_beta
        if smoothness_sim_kernel is not None:
            dirichlet_sim_kernel = smoothness_sim_kernel
        if smoothness_detach_z is not None:
            dirichlet_detach_z = smoothness_detach_z
        self.dirichlet_weight = dirichlet_weight
        self.dirichlet_on = dirichlet_on
        self.dirichlet_mode = dirichlet_mode
        self.dirichlet_dist_alpha = dirichlet_dist_alpha
        self.dirichlet_dist_beta = dirichlet_dist_beta
        self.dirichlet_sim_kernel = dirichlet_sim_kernel
        self.dirichlet_detach_z = dirichlet_detach_z
        # Backward-compatible aliases for historical notebooks.
        self.smoothness_weight = self.dirichlet_weight
        self.smoothness_on = self.dirichlet_on
        self.smoothness_mode = self.dirichlet_mode
        self.smoothness_dist_alpha = self.dirichlet_dist_alpha
        self.smoothness_dist_beta = self.dirichlet_dist_beta
        
        # --------------------------------------------------
        # Optional z-only batch adversarial loss
        # --------------------------------------------------
        self.use_z_adversarial = bool(use_z_adversarial and n_batch > 1)
        self.adv_hidden_dim = adv_hidden_dim
        self.adv_dropout = adv_dropout
        self.adv_on = adv_on

        if adv_on not in ["sample", "mean"]:
            raise ValueError(
                f"Unknown adv_on: {adv_on}. Choose from ['sample', 'mean']."
            )

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        cat_list = [n_batch]

        self.base_encoder = Encoder(
            n_input,
            n_output,
            n_cat_list=cat_list,
            n_layers=2,
            n_hidden=n_hidden,
            dropout_rate=0,
            distribution=self.latent_distribution,
            inject_covariates=False,
            use_batch_norm=False,
            use_layer_norm=False,
            var_activation=None,
            var_eps = var_eps
        )
        if self.spatial_encoder_type == "gat":
            self.gat_mean = GATv2Conv(
                in_channels=n_spatial,
                out_channels=n_spatial,
                add_self_loops=False,
                heads=heads,
                concat=False,
            )
            self.gat_var = GATv2Conv(
                in_channels=n_spatial,
                out_channels=n_spatial,
                add_self_loops=False,
                heads=heads,
                concat=False,
            )
            self.var_head = None

        elif self.spatial_encoder_type == "xattn":
            self.gat_mean = DistanceAwareSpatialCrossAttention(
                in_channels=n_spatial,
                out_channels=n_spatial,
                n_heads=heads,
                distance_aware=distance_aware,
                distance_emb_dim=distance_emb_dim,
                dropout=dropout_rate,
            )
            self.gat_var = None
            self.var_head = nn.Sequential(
                nn.Linear(n_spatial, n_spatial),
                nn.ReLU(),
                nn.Linear(n_spatial, n_spatial),
            )
        else:
            raise ValueError(
                f"Unknown spatial_encoder_type: {self.spatial_encoder_type}. "
                "Choose from ['gat', 'xattn']."
            )

        self.base_encoder2 = Encoder(
            n_spatial,
            n_spatial,
            n_cat_list=cat_list,
            n_layers=1,
            n_hidden=n_spatial,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=False,
            var_activation=None,
            var_eps = var_eps
        )

        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        self.decoder = DecoderSCVI(
            n_output + n_spatial,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        
        # --------------------------------------------------
        # Continuous/protein decoder for real-valued observation models.
        # Important:
        #   DecoderSCVI is designed for count likelihoods and uses
        #   library-size scaling / gene-frequency outputs.
        #   For continuous protein intensity data, we use a raw decoder
        #   that directly reconstructs feature intensities.
        # --------------------------------------------------
        normal_decoder_layers = []
        normal_in_dim = n_output + n_spatial + n_batch

        for _ in range(n_layers):
            normal_decoder_layers.append(nn.Linear(normal_in_dim, n_hidden))
            normal_decoder_layers.append(nn.LayerNorm(n_hidden))
            normal_decoder_layers.append(nn.ReLU())
            normal_decoder_layers.append(nn.Dropout(dropout_rate))
            normal_in_dim = n_hidden

        normal_decoder_layers.append(nn.Linear(normal_in_dim, n_input))
        self.normal_decoder = nn.Sequential(*normal_decoder_layers)

        # --------------------------------------------------
        # Mixture-NB protein decoder (background / foreground), for
        # dis_to_use in {"mixture_nb", "mnb"}. Following scProca/scMuMA:
        # background mean beta, foreground mean alpha*beta (alpha>1), shared
        # per-protein dispersion theta, and a per-cell mixing logit pi.
        # --------------------------------------------------
        if self.dis_to_use in ("mixture_nb", "mnb"):
            mnb_layers = []
            mnb_in_dim = n_output + n_spatial + n_batch
            for _ in range(n_layers):
                mnb_layers.append(nn.Linear(mnb_in_dim, n_hidden))
                mnb_layers.append(nn.LayerNorm(n_hidden))
                mnb_layers.append(nn.ReLU())
                mnb_layers.append(nn.Dropout(dropout_rate))
                mnb_in_dim = n_hidden
            self.mnb_shared = nn.Sequential(*mnb_layers)
            self.mnb_beta = nn.Linear(mnb_in_dim, n_input)   # log background mean
            self.mnb_alpha = nn.Linear(mnb_in_dim, n_input)  # foreground ratio (relu+1 -> >1)
            self.mnb_pi = nn.Linear(mnb_in_dim, n_input)     # mixing logit
            self.mnb_log_theta = nn.Parameter(2.0 * torch.randn(n_input))  # per-protein dispersion

        # --------------------------------------------------
        # Batch discriminator on intrinsic latent z only.
        # This is used only when use_z_adversarial=True and n_batch > 1.
        # --------------------------------------------------
        if self.use_z_adversarial:
            self.classifier_z = nn.Sequential(
                nn.Linear(n_output, adv_hidden_dim),
                nn.LayerNorm(adv_hidden_dim),
                nn.ReLU(),
                nn.Dropout(adv_dropout),
                nn.Linear(adv_hidden_dim, adv_hidden_dim),
                nn.LayerNorm(adv_hidden_dim),
                nn.ReLU(),
                nn.Dropout(adv_dropout),
                nn.Linear(adv_hidden_dim, n_batch),
            )
        else:
            self.classifier_z = None

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        tensors: Dict[str, torch.Tensor], eval_mode = False,
    ) -> Dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(x=x, batch_index=batch_index)
        return input_dict

    def freeze_params(self):
        # freeze
        for param in self.base_encoder.encoder.parameters():
            param.requires_grad = False

        for param in self.gat_mean.parameters():
            param.requires_grad = False

        if self.gat_var is not None:
            for param in self.gat_var.parameters():
                param.requires_grad = False

        if self.var_head is not None:
            for param in self.var_head.parameters():
                param.requires_grad = False
                
        for _, mod in self.base_encoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0    
        #for _, mod in self.decoder.named_modules():
        #    if isinstance(mod, torch.nn.BatchNorm1d):
        #        mod.momentum = 0
        #        mod.affine = False

        


        
        
            

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(edge_index, tuple):
            edge_index, edge_distance = edge_index
        else:
            edge_distance = None

        x_ = x
        if self.dis_to_use in ["zinb", "nb", "poisson"]:
            if self.use_observed_lib_size:
                library = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)
            x_ = torch.log(1 + x_)
        elif self.dis_to_use in CONTINUOUS_LIKELIHOODS:
            # For continuous/log-normalized expression, do not apply another log1p transform.
            # Also avoid count-based library-size scaling.
            if self.use_observed_lib_size:
                library = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            x_ = x_
        elif self.dis_to_use in ("mixture_nb", "mnb"):
            # Protein counts under a two-component mixture-NB likelihood: log1p
            # the encoder input like counts, but the decoder outputs background/
            # foreground means directly, so skip count library-size scaling.
            if self.use_observed_lib_size:
                library = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            x_ = torch.log(1 + x_)
        else:
            raise ValueError(
                f"Unknown dis_to_use: {self.dis_to_use}. "
                f"Choose from {sorted(SUPPORTED_LIKELIHOODS)}."
            )

        q_m, q_v, z = self.base_encoder(x_, batch_index)
        #q = self.base_encoder.encoder(x_, batch_index)
        
        #q_m, q_v, z = self.base_encoder2(q, batch_index)
        spatial_input = q_m[:, -self.n_spatial:]

        if self.spatial_encoder_type == "gat":
            qgat_m = self.gat_mean(spatial_input, edge_index)
            qgat_v = self.gat_var(spatial_input, edge_index)
            qgat_v = torch.exp(qgat_v) + self.var_eps
            attention_dict = None

        elif self.spatial_encoder_type == "xattn":
            qgat_m, attention_dict = self.gat_mean(
                spatial_input,
                edge_index,
                edge_distance=edge_distance,
                return_attention=self.return_attention,
            )

            if self.var_mode == "mlp":
                qgat_v = torch.exp(self.var_head(qgat_m)) + self.var_eps
            else:
                raise ValueError(
                    f"Unknown var_mode: {self.var_mode}. "
                    "Currently supported: ['mlp']."
                )
        
        dist = Normal(qgat_m, qgat_v.sqrt())
        z_gat = dist.rsample()
        
        z_all = torch.cat((z_gat,z),1)
        qall_m = torch.cat((qgat_m,q_m),1)

        #z_all = z_gat @ torch.diag(self.stg(z_gat)) + z
        #qall_m = qgat_m @ torch.diag(self.stg(qgat_m)) + q_m
        #z_all = self.stg(z_gat) + 0.5 * z
        #qall_m = qgat_m @ torch.diag(self.stg.get_gates()) + 0.5 * q_m
        #z_all = z_gat + z
        #qall_m = qgat_m + q_m
        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        if n_samples > 1:
            q_m = self._reshape_tensor_for_samples(q_m, n_samples)
            q_v = self._reshape_tensor_for_samples(q_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qgat_m = self._reshape_tensor_for_samples(qgat_m, n_samples)
            qgat_v = self._reshape_tensor_for_samples(qgat_v, n_samples)
            qall_m = self._reshape_tensor_for_samples(qall_m, n_samples)
            z_gat = self._reshape_tensor_for_samples(z_gat, n_samples)
            z_all = self._reshape_tensor_for_samples(z_all, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(
            z=z,
            q_m=q_m,
            q_v=q_v,
            qgat_m=qgat_m,
            qgat_v=qgat_v,
            z_gat=z_gat,
            qall_m=qall_m,
            z_all=z_all,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
            batch_index=batch_index,
            attention_dict=attention_dict,
        )
        return outputs

    @auto_move_data
    def inference(
        self,
        data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        n_samples: int = 1,
        eval_mode = False,
    ) -> Dict[str, torch.Tensor]:
        inference_input = self._get_inference_input_from_concat_tensors(data,eval_mode)
        outputs = self._generic_inference(**inference_input, edge_index = edge_index, n_samples=n_samples)
        return outputs


    @auto_move_data
    def _generic_generative(
        self,
        z_all: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if self.dis_to_use in CONTINUOUS_LIKELIHOODS:
            # --------------------------------------------------
            # Continuous/protein reconstruction branch.
            # No softmax, no library-size scaling, no count decoder.
            # --------------------------------------------------
            if self.n_batch > 0:
                batch_index_long = batch_index.long().view(-1)
                batch_oh = F.one_hot(
                    batch_index_long,
                    num_classes=self.n_batch,
                ).to(dtype=z_all.dtype, device=z_all.device)

                normal_input = torch.cat([z_all, batch_oh], dim=-1)
            else:
                normal_input = z_all

            px_rate = self.normal_decoder(normal_input)
            if self.continuous_log_scale is None:
                px_scale = None
            else:
                # A per-feature scale is sufficient for the first robust
                # likelihood ablation and adds only n_input parameters. Clamp
                # solely as a numerical safeguard; the proper NLL contains the
                # log-scale term and therefore penalizes variance inflation.
                px_scale = torch.exp(self.continuous_log_scale).clamp(
                    min=1e-4,
                    max=1e4,
                )
            px_dropout = torch.zeros_like(px_rate)
            if self.dis_to_use == "student_t":
                # px_r is the count dispersion in NB branches. In this branch
                # it carries Student-t degrees of freedom so the existing
                # generative-output interface stays checkpoint compatible.
                px_r = torch.full(
                    (self.n_input,),
                    self.student_t_df,
                    device=px_rate.device,
                    dtype=px_rate.dtype,
                )
            else:
                px_r = torch.ones(
                    self.n_input,
                    device=px_rate.device,
                    dtype=px_rate.dtype,
                )

        elif self.dis_to_use in ("mixture_nb", "mnb"):
            # --------------------------------------------------
            # Protein mixture-NB branch (background / foreground).
            # px_scale carries the foreground ratio alpha (>1).
            # --------------------------------------------------
            if self.n_batch > 0:
                batch_index_long = batch_index.long().view(-1)
                batch_oh = F.one_hot(
                    batch_index_long, num_classes=self.n_batch
                ).to(dtype=z_all.dtype, device=z_all.device)
                mnb_input = torch.cat([z_all, batch_oh], dim=-1)
            else:
                mnb_input = z_all
            h = self.mnb_shared(mnb_input)
            beta = torch.exp(torch.clamp(self.mnb_beta(h), max=12.0))   # background mean (>0)
            alpha = F.relu(self.mnb_alpha(h)) + 1.0 + 1e-8              # foreground ratio (>1)
            pi = self.mnb_pi(h)                                         # mixing logit
            px_rate = beta
            px_scale = alpha
            px_dropout = pi
            px_r = torch.exp(self.mnb_log_theta)

        else:
            # --------------------------------------------------
            # Count-based RNA branch.
            # Uses scVI-style decoder.
            # --------------------------------------------------
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                z_all,
                library,
                batch_index,
            )
            px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
        )

    @auto_move_data
    def generative(
        self,
        latent: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = latent["z_all"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        latent_batch_size = latent["z"].shape[batch_size_dim]
        generative_input = {}

        for key in ["z_all", "library","batch_index"]:
            generative_input[key] = latent[key]

        outputs = self._generic_generative(**generative_input)
        return outputs

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        dis_to_use: str = "zinb",
        px_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruction loss under different observation models.
        Supported:
            - "zinb"
            - "nb"
            - "poisson"
            - "normal"
            - "normal_feature_scale"
            - "student_t"
            - "mixture_nb" / "mnb"
        """
        if dis_to_use == "zinb":
            recon_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate,
                    theta=px_r,
                    zi_logits=px_dropout,
                )
                .log_prob(x)
                .sum(dim=-1)
            )

        elif dis_to_use == "nb":
            recon_loss = (
                -NegativeBinomial(
                    mu=px_rate,
                    theta=px_r,
                )
                .log_prob(x)
                .sum(dim=-1)
            )

        elif dis_to_use == "poisson":
            recon_loss = (
                -Poisson(rate=px_rate.clamp_min(1e-8))
                .log_prob(x)
                .sum(dim=-1)
            )

        elif dis_to_use == "normal":
            # Historical behavior retained exactly for reproducibility. This
            # is proportional to a fixed-variance Gaussian NLL, but omits its
            # constant and 1/2 factor.
            recon_loss = F.mse_loss(
                px_rate,
                x,
                reduction="none",
            ).sum(dim=-1)

        elif dis_to_use == "normal_feature_scale":
            if px_scale is None:
                raise ValueError(
                    "px_scale is required for normal_feature_scale."
                )
            recon_loss = (
                -Normal(loc=px_rate, scale=px_scale)
                .log_prob(x)
                .sum(dim=-1)
            )

        elif dis_to_use == "student_t":
            if px_scale is None:
                raise ValueError("px_scale is required for student_t.")
            recon_loss = (
                -StudentT(df=px_r, loc=px_rate, scale=px_scale)
                .log_prob(x)
                .sum(dim=-1)
            )

        elif dis_to_use in ("mixture_nb", "mnb"):
            recon_loss = mixture_nb_loss(
                x,
                beta=px_rate,
                alpha=px_scale,
                pi=px_dropout,
                theta=px_r,
            )

        else:
            raise ValueError(
                f"Unknown dis_to_use: {dis_to_use}. "
                f"Choose from {sorted(SUPPORTED_LIKELIHOODS)}."
            )

        return recon_loss

    def get_z_for_adversarial(
        self,
        inference_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Select intrinsic z representation for batch adversarial training.

        adv_on="sample": use sampled z, consistent with scMuMA.
        adv_on="mean": use posterior mean q_m.
        """
        if self.adv_on == "sample":
            return inference_outputs["z"]
        elif self.adv_on == "mean":
            return inference_outputs["q_m"]
        else:
            raise ValueError(
                f"Unknown adv_on: {self.adv_on}. Choose from ['sample', 'mean']."
            )

    def classifier_loss_z(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss for predicting batch labels from intrinsic z.
        Used for z-only adversarial batch correction.
        """
        if not self.use_z_adversarial or self.classifier_z is None:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        logits = self.classifier_z(z)
        target = batch_index.long().view(-1)

        return F.cross_entropy(logits, target, reduction="mean")

    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _generic_loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]



        q_m = inference_outputs["q_m"]
        q_v = inference_outputs["q_v"]
        qgat_m = inference_outputs["qgat_m"]
        qgat_v = inference_outputs["qgat_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        px_scale = generative_outputs.get("px_scale", None)

        prior_z_m = torch.zeros_like(q_m)
        prior_z_v = torch.ones_like(q_v)
        prior_zgat_m = torch.zeros_like(qgat_m)
        prior_zgat_v = torch.ones_like(qgat_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout, self.dis_to_use, px_scale=px_scale)
        if self.reg_to_use == 'mmd':
            mi_loss = self.corr_loss(inference_outputs)
        else:
            mi_loss = self.mi_loss(inference_outputs)
        kl_z = self.latent_kl_divergence(q_m, q_v, prior_z_m, prior_z_v)
        kl_zgat = self.latent_kl_divergence(qgat_m, qgat_v, prior_zgat_m, prior_zgat_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        if self.dirichlet_weight > 0 and edge_index is not None:
            dirichlet_energy_loss = self.dirichlet_energy_loss(inference_outputs, edge_index)
        else:
            dirichlet_energy_loss = torch.tensor(0.0, device=q_m.device)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            kl_zgat=kl_zgat,
            kl_library=kl_library,
            mi_loss = mi_loss,
            dirichlet_energy_loss=dirichlet_energy_loss,
            smoothness_loss=dirichlet_energy_loss,
        )

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
        weight = 1,
        edge_index: Optional[torch.Tensor] = None,
        adversarial_weight: float = 0.0,
    ) -> LossOutput:
        """
        Compute (generator) loss terms for NicheEnergy. 

        """

        losses = self._generic_loss(
            tensors,
            inference_outputs,
            generative_outputs,
            edge_index=edge_index,
        )
        reconst_loss = losses["recon_loss"]
        mi_loss = losses["mi_loss"]
        kl_divergence_z = losses["kl_z"]
        kl_divergence_zgat = losses["kl_zgat"]
        kl_divergence_l = losses["kl_library"]
        dirichlet_energy_loss = losses["dirichlet_energy_loss"]

        kl_local_for_warmup = kl_divergence_z + kl_divergence_zgat
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = self.kl_weight * kl_divergence_z + self.kl_s_weight * kl_divergence_zgat + kl_local_no_warmup

        loss = (
            torch.mean(reconst_loss + weight * weighted_kl_local)
            + weight * self.lam_mi * mi_loss
            + self.dirichlet_weight * dirichlet_energy_loss
        )

        # --------------------------------------------------
        # z-only adversarial batch loss.
        # This term is negative for the generator/encoder update:
        # classifier is trained separately to minimize CE,
        # while encoder is trained to maximize CE and fool classifier.
        # --------------------------------------------------
        if self.use_z_adversarial and adversarial_weight > 0:
            z_for_adv = self.get_z_for_adversarial(inference_outputs)
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            loss_adv_z = -self.classifier_loss_z(z_for_adv, batch_index)
            loss = loss + adversarial_weight * loss_adv_z
            
        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_zgat=kl_divergence_zgat
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=self.lam_mi * mi_loss,
            kl_global=kl_local,
        )
        
    def mi_loss(
        self,
        inference_outputs: Dict[str, torch.Tensor],
    ):
        """
        Compute MI loss terms for NicheEnergy. 
        """
        psi_x = inference_outputs["z"]
        psi_y = inference_outputs["z_gat"].detach()
        
        C_yy = self._cov(psi_y, psi_y)
        C_yx = self._cov(psi_y, psi_x)
        C_xy = self._cov(psi_x, psi_y)
        C_xx = self._cov(psi_x, psi_x)

        C_xx_inv = torch.inverse(C_xx+torch.eye(C_xx.shape[0], device=psi_x.device)*1e-3)

        l2 = -torch.logdet(C_yy-torch.linalg.multi_dot([C_yx,C_xx_inv,C_xy])) + torch.logdet(C_yy)
        return l2
    
    
    @staticmethod
    def _cosine_similarity_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
        x = F.normalize(x, p=2, dim=1, eps=eps)
        y = F.normalize(y, p=2, dim=1, eps=eps)
        cos_sim = (x * y).sum(dim=1)
        return 0.5 * (1.0 + cos_sim)

    def _sim_kernel(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
        """Edge intrinsic-similarity kernel for the Dirichlet weight.
        half_cos: 0.5(1+cos) in [0,1] (default, historical) -> orthogonal z gives 0.5.
        relu:     max(0, cos) -> vanishes for orthogonal/anti-aligned z (true boundary insulation).
        exp:      exp((cos-1)/tau), tau=0.5 -> smooth sharpening toward the same limit."""
        xn = F.normalize(x, p=2, dim=1, eps=eps)
        yn = F.normalize(y, p=2, dim=1, eps=eps)
        cos = (xn * yn).sum(dim=1)
        mode = getattr(self, "dirichlet_sim_kernel", "half_cos")
        if mode == "half_cos":
            return 0.5 * (1.0 + cos)
        if mode == "relu":
            return torch.clamp(cos, min=0.0)
        if mode == "pow2":                       # sharper but smoother than relu (no hard zero)
            return (0.5 * (1.0 + cos)) ** 2
        if mode == "pow4":
            return (0.5 * (1.0 + cos)) ** 4
        if mode == "exp":
            return torch.exp((cos - 1.0) / 0.5)
        raise ValueError(f"Unknown dirichlet_sim_kernel: {mode}")

    def _distance_weight(self, edge_distance: torch.Tensor):
        """
        Distance weight for Dirichlet energy regularization.

        Given edge distances d:
            d_norm = d / median(d)

        We define:
            base_w = 1 / (1 + alpha * d_norm)
            w = (1 - beta) + beta * base_w

        where:
            alpha controls how fast distance decays,
            beta controls how strongly distance affects the final edge weight.

        When beta=0, distance has no effect and w=1.
        When beta is larger, distance plays a stronger role.
        """
        d = edge_distance.float().clamp_min(0.0)

        # robust global scale
        d_scale = torch.median(d).clamp_min(1e-8)
        d_norm = d / d_scale

        # mild distance effect
        alpha = self.dirichlet_dist_alpha
        base_w = 1.0 / (1.0 + alpha * d_norm)

        # shrink toward 1 so distance does not dominate
        beta = self.dirichlet_dist_beta
        w = (1.0 - beta) + beta * base_w

        return w
      
      
    def dirichlet_energy_loss(
        self,
        inference_outputs: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
    ):
        """
        State-modulated Dirichlet energy on posterior mean niche states.

        Loss:
            sum_{(i,j) in E} w_ij * sim(z_i, z_j) * ||s_i - s_j||^2
            --------------------------------------------------------
                     sum_{(i,j) in E} w_ij * sim(z_i, z_j) + eps

        where
            - s is the posterior mean niche state qgat_m
            - z is the posterior mean intrinsic state q_m
            - sim(z_i, z_j) = (1 + cosine(z_i, z_j)) / 2
            - w_ij is a distance-based edge weight
        """
        if self.dirichlet_on == "mean":
            s = inference_outputs["qgat_m"]
        else:
            raise ValueError(
                f"Unknown dirichlet_on: {self.dirichlet_on}. "
                "Currently supported: ['mean']."
            )

        # intrinsic mean representation
        z_mean = inference_outputs["q_m"]
        if getattr(self, "dirichlet_detach_z", False):
            z_mean = z_mean.detach()

        if isinstance(edge_index, tuple):
            edge_index_tensor, edge_distance = edge_index
        else:
            edge_index_tensor = edge_index
            edge_distance = None

        # Optionally recompute the energy field with deterministic attention.
        # Weight detachment and encoder detachment control separate gradient paths.
        _recompute = getattr(self, "dirichlet_energy_recompute", False) or getattr(self, "dirichlet_encoder_detach", False)
        if _recompute:
            spatial_input = inference_outputs["q_m"][:, -self.n_spatial:]
            if getattr(self, "dirichlet_encoder_detach", False):
                spatial_input = spatial_input.detach()
            # Disable attention dropout for this call without changing module state.
            if self.spatial_encoder_type == "xattn":
                s, _ = self.gat_mean(spatial_input, edge_index_tensor,
                                     edge_distance=edge_distance, return_attention=False,
                                     apply_dropout=False)
            else:
                # GATv2 has no per-call dropout override.
                s = self.gat_mean(spatial_input, edge_index_tensor)

        src = edge_index_tensor[0]
        dst = edge_index_tensor[1]

        # spatial difference term
        s_diff_sq = (s[src] - s[dst]).pow(2).sum(dim=1)   # (E,)

        # intrinsic similarity term
        z_sim = self._sim_kernel(z_mean[src], z_mean[dst])   # (E,)

        # distance weight term
        if edge_distance is not None:
            dist_w = self._distance_weight(edge_distance)   # (E,)
        else:
            dist_w = torch.ones_like(z_sim)

        if self.dirichlet_mode == "plain":
            edge_w = dist_w
        elif self.dirichlet_mode == "intrinsic_aware":
            edge_w = dist_w * z_sim
        else:
            raise ValueError(
                f"Unknown dirichlet_mode: {self.dirichlet_mode}. "
                "Currently supported: ['plain', 'intrinsic_aware']."
            )

        denom = edge_w.sum().clamp_min(1e-8)
        loss = (edge_w * s_diff_sq).sum() / denom
        return loss

    def spatial_smoothness_loss(
        self,
        inference_outputs: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
    ):
        """Backward-compatible alias for :meth:`dirichlet_energy_loss`."""
        return self.dirichlet_energy_loss(inference_outputs, edge_index)
      
      
    def corr_loss(
        self,
        inference_outputs: Dict[str, torch.Tensor],
    ):
        """
        Compute MMD loss terms for NicheEnergy. 
        """
        z = inference_outputs["z"]
        s = inference_outputs["z_gat"].detach()

        # The historical RBF-MMD implementation is quadratic in the number of
        # cells.  Existing runs are unchanged because the cap is opt-in.  New
        # large, full-graph analyses may set ``module.mmd_max_samples`` to use
        # a seeded Monte-Carlo estimate of the same objective while retaining
        # full-data reconstruction and graph/Dirichlet losses.
        max_samples = getattr(self, "mmd_max_samples", None)
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < 2:
                raise ValueError("mmd_max_samples must be at least two")
            if z.shape[0] > max_samples:
                indices = torch.randperm(z.shape[0], device=z.device)[:max_samples]
                z = z[indices]
                s = s[indices]
        sample = torch.cat((z,s),1)
        true_samples = Variable(torch.randn(sample.shape[0], sample.shape[1],device=z.device),requires_grad=False)
        l1 = self.compute_mmd(sample,true_samples)

        return l1
    
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
    
    @staticmethod
    def compute_kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    
    @staticmethod
    def _cov(psi_x, psi_y):
        """
        :return: covariance matrix
        """
        N = psi_x.shape[0]
        return (psi_y.T @ psi_x).T / (N - 1)
        
