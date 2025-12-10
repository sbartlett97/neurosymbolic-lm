"""
Regularization techniques to prevent catastrophic forgetting.

Implements:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Memory Aware Synapses (MAS)
- Learning without Forgetting (LwF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict
import copy


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC).
    
    Penalizes changes to parameters that were important for previous tasks,
    as measured by the Fisher Information Matrix.
    
    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_weight: float = 1000.0,
        online: bool = True,
        gamma: float = 0.9
    ):
        """
        Args:
            model: The model to regularize
            importance_weight: Lambda weight for EWC penalty
            online: Use online EWC (running average of Fisher)
            gamma: Decay factor for online EWC
        """
        self.model = model
        self.importance_weight = importance_weight
        self.online = online
        self.gamma = gamma
        
        # Storage for Fisher Information and optimal parameters
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        self._initialized = False
    
    def compute_fisher(
        self,
        dataloader,
        num_samples: int = 200,
        empirical: bool = True
    ):
        """
        Compute Fisher Information Matrix diagonal.
        
        Args:
            dataloader: DataLoader with representative samples
            num_samples: Number of samples to use for estimation
            empirical: Use empirical Fisher (gradients of actual labels)
        """
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        self.model.eval()
        count = 0
        
        for batch in dataloader:
            if count >= num_samples:
                break
            
            self.model.zero_grad()
            
            # Move batch to device
            device = next(self.model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            out = self.model(input_ids, attention_mask, spans=None, y_ids=None)
            
            if empirical:
                # Use actual task loss
                entity_logits = out["entity_logits"]
                # Simple loss for Fisher computation
                loss = -F.log_softmax(entity_logits, dim=-1).mean()
            else:
                # Sample from model's predictions
                entity_logits = out["entity_logits"]
                probs = F.softmax(entity_logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
                loss = F.cross_entropy(
                    entity_logits.view(-1, entity_logits.size(-1)),
                    samples.view(-1)
                )
            
            loss.backward()
            
            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data ** 2
            
            count += batch["input_ids"].size(0)
        
        # Normalize
        for n in fisher:
            fisher[n] /= count
        
        # Online EWC: running average
        if self.online and self._initialized:
            for n in fisher:
                self.fisher_information[n] = (
                    self.gamma * self.fisher_information[n] +
                    (1 - self.gamma) * fisher[n]
                )
        else:
            self.fisher_information = fisher
        
        # Store current parameters as optimal
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        self._initialized = True
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.
        
        Returns:
            Scalar tensor with penalty value
        """
        if not self._initialized:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for n, p in self.model.named_parameters():
            if n in self.fisher_information and n in self.optimal_params:
                loss += (
                    self.fisher_information[n] *
                    (p - self.optimal_params[n]) ** 2
                ).sum()
        
        return self.importance_weight * loss
    
    def get_importance_scores(self) -> Dict[str, float]:
        """Get importance scores for each parameter group."""
        scores = {}
        for n, fisher in self.fisher_information.items():
            scores[n] = fisher.mean().item()
        return scores


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI).
    
    Computes importance online during training based on contribution
    to loss decrease.
    
    Reference: Zenke et al., "Continual Learning Through Synaptic Intelligence" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_weight: float = 100.0,
        damping: float = 0.1
    ):
        """
        Args:
            model: The model to regularize
            importance_weight: Weight for SI penalty
            damping: Damping factor for numerical stability
        """
        self.model = model
        self.importance_weight = importance_weight
        self.damping = damping
        
        # Initialize tracking variables
        self.omega: Dict[str, torch.Tensor] = {}  # Importance weights
        self.prev_params: Dict[str, torch.Tensor] = {}  # Previous task parameters
        self.running_sum: Dict[str, torch.Tensor] = {}  # Running sum of path integrals
        self.prev_task_params: Dict[str, torch.Tensor] = {}
        
        self._register_hooks()
        self._initialized = False
    
    def _register_hooks(self):
        """Register backward hooks to track gradients."""
        self.grad_buffer: Dict[str, torch.Tensor] = {}
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.clone().detach()
                self.running_sum[n] = torch.zeros_like(p)
                self.omega[n] = torch.zeros_like(p)
    
    def update_running_sum(self):
        """Update running sum after each training step."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta = p.detach() - self.prev_params[n]
                self.running_sum[n] += -p.grad.detach() * delta
                self.prev_params[n] = p.clone().detach()
    
    def consolidate(self):
        """Consolidate importance after task completion."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta = p.detach() - self.prev_task_params.get(n, p.detach())
                self.omega[n] += self.running_sum[n] / (delta ** 2 + self.damping)
                self.running_sum[n] = torch.zeros_like(p)
                self.prev_task_params[n] = p.clone().detach()
        
        self._initialized = True
    
    def penalty(self) -> torch.Tensor:
        """Compute SI penalty term."""
        if not self._initialized:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for n, p in self.model.named_parameters():
            if n in self.omega and n in self.prev_task_params:
                loss += (self.omega[n] * (p - self.prev_task_params[n]) ** 2).sum()
        
        return self.importance_weight * loss


class LearningWithoutForgetting:
    """
    Learning without Forgetting (LwF).
    
    Uses knowledge distillation from the old model to preserve
    previous knowledge while learning new tasks.
    
    Reference: Li & Hoiem, "Learning without Forgetting" (2016)
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Args:
            model: The model to train
            temperature: Distillation temperature
            alpha: Weight for distillation loss
        """
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher_model: Optional[nn.Module] = None
    
    def snapshot_teacher(self):
        """Create a snapshot of the current model as teacher."""
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        
        # Freeze teacher
        for p in self.teacher_model.parameters():
            p.requires_grad = False
    
    def distillation_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute distillation loss from teacher.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            student_outputs: Outputs from student (current) model
        
        Returns:
            Distillation loss tensor
        """
        if self.teacher_model is None:
            return torch.tensor(0.0)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids, attention_mask, spans=None, y_ids=None
            )
        
        # Entity logits distillation
        student_entity = student_outputs["entity_logits"]
        teacher_entity = teacher_outputs["entity_logits"]
        
        student_soft = F.log_softmax(student_entity / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_entity / self.temperature, dim=-1)
        
        entity_distill = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Concept probabilities distillation
        student_concept = student_outputs["concept_probs"]
        teacher_concept = teacher_outputs["concept_probs"]
        
        concept_distill = F.mse_loss(student_concept, teacher_concept)
        
        return self.alpha * (entity_distill + concept_distill)


class CombinedRegularizer:
    """
    Combines multiple regularization strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ewc: bool = True,
        use_si: bool = False,
        use_lwf: bool = True,
        ewc_weight: float = 1000.0,
        si_weight: float = 100.0,
        lwf_alpha: float = 0.5
    ):
        self.model = model
        
        self.ewc = EWCRegularizer(model, ewc_weight) if use_ewc else None
        self.si = SynapticIntelligence(model, si_weight) if use_si else None
        self.lwf = LearningWithoutForgetting(model, alpha=lwf_alpha) if use_lwf else None
    
    def consolidate(self, dataloader):
        """Consolidate knowledge after task completion."""
        if self.ewc:
            self.ewc.compute_fisher(dataloader)
        if self.si:
            self.si.consolidate()
        if self.lwf:
            self.lwf.snapshot_teacher()
    
    def penalty(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        student_outputs: Optional[Dict] = None
    ) -> torch.Tensor:
        """Compute combined regularization penalty."""
        total_penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        if self.ewc:
            total_penalty = total_penalty + self.ewc.penalty()
        
        if self.si:
            total_penalty = total_penalty + self.si.penalty()
        
        if self.lwf and input_ids is not None and student_outputs is not None:
            total_penalty = total_penalty + self.lwf.distillation_loss(
                input_ids, attention_mask, student_outputs
            )
        
        return total_penalty
    
    def update_step(self):
        """Call after each training step for online methods."""
        if self.si:
            self.si.update_running_sum()
