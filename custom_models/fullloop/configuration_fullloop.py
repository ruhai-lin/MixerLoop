from custom_models.mixerloop import MixerLoopConfig


class FullLoopConfig(MixerLoopConfig):
    """Repeat the entire GDN + FFN stack, with LT2's inter-iteration residual."""

    model_type = 'fullloop'

    def __init__(
        self,
        mixer_norm_eps: float | None = None,
        allow_neg_eigval: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('initializer_range', kwargs.get('hidden_size', 256) ** -0.5)
        super().__init__(**kwargs)
        self.mixer_norm_eps = self.norm_eps if mixer_norm_eps is None else mixer_norm_eps
        self.allow_neg_eigval = allow_neg_eigval
