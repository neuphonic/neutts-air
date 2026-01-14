from neutts.neutts import NeuTTS


class NeuTTSAir(NeuTTS):
    """
    NeuTTSAir is a subclass of NeuTTS.
    It inherits all methods and attributes automatically.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
