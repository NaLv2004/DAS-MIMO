from .codec import LDPC5GCodec
from .decoder import OffsetMinSumDecoder


def build_decoder_aware_codec(config):
    return LDPC5GCodec(
        config.DECODER_AWARE_CODEWORD_LENGTH,
        config.DECODER_AWARE_CODE_RATE,
        device=config.DEVICE,
        rv=0,
        qm=1,
        nlayers=1,
    )


def build_decoder_aware_decoder(config, codec):
    return OffsetMinSumDecoder(
        codec.params, num_iterations=config.DECODER_AWARE_DECODER_ITERS, offset=0.25
    )
