"""
Uncertainty estimation for identifying what the model doesn't know.

Implements multiple uncertainty quantification methods:
- Monte Carlo Dropout
- Ensemble disagreement
- Predictive entropy
- Epistemic vs Aleatoric uncertainty decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty metrics."""
    predictive_entropy: float
    epistemic_uncertainty: float  # Model uncertainty (reducible)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    entity_uncertainty: float
    concept_uncertainty: float
    relation_uncertainty: float
    overall_score: float  # Combined uncertainty score
    
    def is_uncertain(self, threshold: float = 0.5) -> bool:
        """Check if overall uncertainty exceeds threshold."""
        return self.overall_score > threshold
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "predictive_entropy": self.predictive_entropy,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "entity_uncertainty": self.entity_uncertainty,
            "concept_uncertainty": self.concept_uncertainty,
            "relation_uncertainty": self.relation_uncertainty,
            "overall_score": self.overall_score
        }


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled to estimate
    predictive uncertainty through sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: Optional[float] = None
    ):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC samples
            dropout_rate: Override dropout rate (None uses model's default)
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
    
    def _restore_eval(self):
        """Restore model to eval mode."""
        self.model.eval()
    
    @torch.no_grad()
    def estimate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> UncertaintyMetrics:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
        
        Returns:
            UncertaintyMetrics with various uncertainty measures
        """
        was_training = self.model.training
        
        # Collect predictions from multiple forward passes
        all_entity_probs = []
        all_concept_probs = []
        all_relation_probs = []
        all_decoder_probs = []
        
        for _ in range(self.num_samples):
            self._enable_dropout()
            
            out = self.model(input_ids, attention_mask, spans=None, y_ids=None)
            
            # Entity probabilities
            entity_probs = F.softmax(out["entity_logits"], dim=-1)
            all_entity_probs.append(entity_probs.cpu())
            
            # Concept probabilities
            concept_probs = out["concept_probs"]
            all_concept_probs.append(concept_probs.cpu())
            
            # Relation probabilities from matrix
            if "rel_logits_matrix" in out:
                rel_probs = F.softmax(out["rel_logits_matrix"], dim=-1)
                all_relation_probs.append(rel_probs.cpu())
        
        self._restore_eval()
        if was_training:
            self.model.train()
        
        # Compute uncertainty metrics
        metrics = self._compute_metrics(
            all_entity_probs,
            all_concept_probs,
            all_relation_probs
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        entity_probs: List[torch.Tensor],
        concept_probs: List[torch.Tensor],
        relation_probs: List[torch.Tensor]
    ) -> UncertaintyMetrics:
        """Compute uncertainty metrics from collected predictions."""
        
        # Stack predictions: (num_samples, B, ...)
        entity_stack = torch.stack(entity_probs, dim=0)
        concept_stack = torch.stack(concept_probs, dim=0)
        
        # Entity uncertainty
        entity_mean = entity_stack.mean(dim=0)
        entity_variance = entity_stack.var(dim=0)
        entity_entropy = self._entropy(entity_mean)
        entity_uncertainty = entity_variance.mean().item()
        
        # Concept uncertainty
        concept_mean = concept_stack.mean(dim=0)
        concept_variance = concept_stack.var(dim=0)
        concept_uncertainty = concept_variance.mean().item()
        
        # Relation uncertainty
        relation_uncertainty = 0.0
        if relation_probs:
            relation_stack = torch.stack(relation_probs, dim=0)
            relation_variance = relation_stack.var(dim=0)
            relation_uncertainty = relation_variance.mean().item()
        
        # Predictive entropy (total uncertainty)
        predictive_entropy = entity_entropy.mean().item()
        
        # Expected entropy (aleatoric - irreducible)
        individual_entropies = torch.stack([self._entropy(p) for p in entity_probs])
        expected_entropy = individual_entropies.mean(dim=0)
        aleatoric = expected_entropy.mean().item()
        
        # Epistemic uncertainty (mutual information - reducible)
        epistemic = predictive_entropy - aleatoric
        
        # Combined score (weighted average)
        overall = 0.4 * epistemic + 0.3 * entity_uncertainty + 0.2 * concept_uncertainty + 0.1 * relation_uncertainty
        
        return UncertaintyMetrics(
            predictive_entropy=predictive_entropy,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            entity_uncertainty=entity_uncertainty,
            concept_uncertainty=concept_uncertainty,
            relation_uncertainty=relation_uncertainty,
            overall_score=overall
        )
    
    def _entropy(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=dim)


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty estimation.
    
    Uses multiple model checkpoints to estimate uncertainty through disagreement.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of model instances (different checkpoints/initializations)
        """
        self.models = models
    
    @torch.no_grad()
    def estimate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> UncertaintyMetrics:
        """Estimate uncertainty using ensemble disagreement."""
        all_entity_probs = []
        all_concept_probs = []
        
        for model in self.models:
            model.eval()
            out = model(input_ids, attention_mask, spans=None, y_ids=None)
            
            entity_probs = F.softmax(out["entity_logits"], dim=-1)
            all_entity_probs.append(entity_probs.cpu())
            all_concept_probs.append(out["concept_probs"].cpu())
        
        # Compute disagreement
        entity_stack = torch.stack(all_entity_probs, dim=0)
        concept_stack = torch.stack(all_concept_probs, dim=0)
        
        # Variance as disagreement measure
        entity_disagreement = entity_stack.var(dim=0).mean().item()
        concept_disagreement = concept_stack.var(dim=0).mean().item()
        
        # Mean predictions
        entity_mean = entity_stack.mean(dim=0)
        predictive_entropy = -torch.sum(entity_mean * torch.log(entity_mean + 1e-10), dim=-1).mean().item()
        
        return UncertaintyMetrics(
            predictive_entropy=predictive_entropy,
            epistemic_uncertainty=entity_disagreement,
            aleatoric_uncertainty=0.0,  # Not separable with ensemble
            entity_uncertainty=entity_disagreement,
            concept_uncertainty=concept_disagreement,
            relation_uncertainty=0.0,
            overall_score=entity_disagreement
        )


class UncertaintyEstimator:
    """
    Unified uncertainty estimation interface.
    
    Combines multiple methods for robust uncertainty quantification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "mc_dropout",
        num_samples: int = 10,
        ensemble_models: Optional[List[nn.Module]] = None
    ):
        """
        Args:
            model: Primary model
            method: Estimation method ('mc_dropout', 'ensemble', 'combined')
            num_samples: Number of MC samples
            ensemble_models: Additional models for ensemble
        """
        self.model = model
        self.method = method
        
        self.mc_dropout = MonteCarloDropout(model, num_samples)
        
        if ensemble_models:
            self.ensemble = EnsembleUncertainty([model] + ensemble_models)
        else:
            self.ensemble = None
    
    def estimate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> UncertaintyMetrics:
        """
        Estimate uncertainty for given input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            UncertaintyMetrics
        """
        if self.method == "mc_dropout":
            return self.mc_dropout.estimate(input_ids, attention_mask)
        elif self.method == "ensemble" and self.ensemble:
            return self.ensemble.estimate(input_ids, attention_mask)
        elif self.method == "combined" and self.ensemble:
            # Average both methods
            mc_metrics = self.mc_dropout.estimate(input_ids, attention_mask)
            ens_metrics = self.ensemble.estimate(input_ids, attention_mask)
            
            return UncertaintyMetrics(
                predictive_entropy=(mc_metrics.predictive_entropy + ens_metrics.predictive_entropy) / 2,
                epistemic_uncertainty=(mc_metrics.epistemic_uncertainty + ens_metrics.epistemic_uncertainty) / 2,
                aleatoric_uncertainty=mc_metrics.aleatoric_uncertainty,
                entity_uncertainty=(mc_metrics.entity_uncertainty + ens_metrics.entity_uncertainty) / 2,
                concept_uncertainty=(mc_metrics.concept_uncertainty + ens_metrics.concept_uncertainty) / 2,
                relation_uncertainty=mc_metrics.relation_uncertainty,
                overall_score=(mc_metrics.overall_score + ens_metrics.overall_score) / 2
            )
        else:
            return self.mc_dropout.estimate(input_ids, attention_mask)
    
    def batch_estimate(
        self,
        samples: List[Dict],
        tokenizer,
        device: str = "cpu"
    ) -> List[Tuple[Dict, UncertaintyMetrics]]:
        """
        Estimate uncertainty for a batch of samples.
        
        Args:
            samples: List of sample dictionaries with 'text' key
            tokenizer: Tokenizer for encoding
            device: Device for computation
        
        Returns:
            List of (sample, uncertainty) tuples sorted by uncertainty (highest first)
        """
        results = []
        
        for sample in samples:
            tokenized = tokenizer(
                sample["text"],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            uncertainty = self.estimate(input_ids, attention_mask)
            results.append((sample, uncertainty))
        
        # Sort by uncertainty (highest first)
        results.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return results
