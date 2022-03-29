
import logging
from fairseq.models.bart import BARTModel

from ..modules.output_heads import SentenceClassificationHead

logger = logging.getLogger(__name__)


def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
):
    """Register a classification head."""
    logger.info("Registering classification head: {0}".format(name))
    if name in self.classification_heads:
        prev_num_classes = self.classification_heads[name].out_proj.out_features
        prev_inner_dim = self.classification_heads[name].dense.out_features
        if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
            logger.warning(
                're-registering head "{}" with num_classes {} (prev: {}) '
                "and inner_dim {} (prev: {})".format(
                    name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                )
            )

    if 'gcn' in name:
        pass
    else:
        self.classification_heads[name] = SentenceClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            pooler_activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )



BARTModel.register_classification_head = register_classification_head