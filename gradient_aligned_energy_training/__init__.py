# Import the loss module so AllenNLP's --include-package triggers registration.
from gradient_aligned_energy_training.gaet_loss import GradientAlignedLoss

# Also ensure MLME energy is importable (registered via seal package).
