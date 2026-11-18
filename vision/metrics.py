import torch


def compute_binary_purity(activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	"""Compute binary purity for each latent dimension.

	Binary purity for latent dimension j is defined as:
		max_class_count / number_of_activating_samples

	where a sample is considered activating if activations[i, j] > 0.

	Args:
		activations: Tensor of shape (num_images, num_latents).
		labels: Tensor of shape (num_images,) with integer class labels.

	Returns:
		Tensor of shape (num_latents,) containing purity scores in [0, 1].
		Latents with no positive activations receive purity 0.
	"""
	activations = torch.as_tensor(activations)
	labels = torch.as_tensor(labels)

	if activations.ndim != 2:
		raise ValueError(
			f"Expected activations to have shape (num_images, num_latents), got {tuple(activations.shape)}"
		)
	if labels.ndim != 1:
		raise ValueError(f"Expected labels to have shape (num_images,), got {tuple(labels.shape)}")
	if activations.shape[0] != labels.shape[0]:
		raise ValueError(
			"activations and labels must have the same number of samples: "
			f"got {activations.shape[0]} and {labels.shape[0]}"
		)

	if labels.numel() == 0:
		return torch.zeros(activations.shape[1], dtype=torch.float32, device=activations.device)

	if not torch.is_floating_point(activations):
		activations = activations.float()
	labels = labels.long().to(activations.device)

	num_latents = activations.shape[1]
	active_mask = activations > 0
	active_indices = active_mask.nonzero(as_tuple=False)

	if active_indices.numel() == 0:
		return torch.zeros(num_latents, dtype=torch.float32, device=activations.device)

	active_samples = active_indices[:, 0]
	active_latents = active_indices[:, 1]
	active_labels = labels[active_samples]

	num_classes = int(labels.max().item()) + 1
	flat_indices = active_latents * num_classes + active_labels
	class_counts = torch.bincount(flat_indices, minlength=num_latents * num_classes).view(
		num_latents, num_classes
	)

	max_class_counts = class_counts.max(dim=1).values.float()
	active_counts = torch.bincount(active_latents, minlength=num_latents).float()
	purity = torch.where(
		active_counts > 0,
		max_class_counts / active_counts,
		torch.zeros(num_latents, dtype=torch.float32, device=activations.device),
	)

	return purity


def compute_multilabel_binary_purity(activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	"""Compute latent purity for multi-label targets.

	For latent j, purity is the max class support among activating samples:
		max_c mean(labels[active_for_j, c])

	Args:
		activations: Tensor of shape (num_images, num_latents).
		labels: Multi-hot tensor of shape (num_images, num_classes).

	Returns:
		Tensor of shape (num_latents,) with values in [0, 1].
	"""
	activations = torch.as_tensor(activations)
	labels = torch.as_tensor(labels)

	if activations.ndim != 2:
		raise ValueError(
			f"Expected activations to have shape (num_images, num_latents), got {tuple(activations.shape)}"
		)
	if labels.ndim != 2:
		raise ValueError(f"Expected labels to have shape (num_images, num_classes), got {tuple(labels.shape)}")
	if activations.shape[0] != labels.shape[0]:
		raise ValueError(
			"activations and labels must have the same number of samples: "
			f"got {activations.shape[0]} and {labels.shape[0]}"
		)

	if labels.numel() == 0:
		return torch.zeros(activations.shape[1], dtype=torch.float32, device=activations.device)

	if not torch.is_floating_point(activations):
		activations = activations.float()
	labels = labels.float().to(activations.device)

	num_latents = activations.shape[1]
	active_mask = activations > 0
	active_counts = active_mask.sum(dim=0).float()

	class_support = active_mask.float().T @ labels
	class_support = class_support / (active_counts.unsqueeze(1) + 1e-12)
	max_support = class_support.max(dim=1).values

	purity = torch.where(
		active_counts > 0,
		max_support,
		torch.zeros(num_latents, dtype=torch.float32, device=activations.device),
	)

	return purity
