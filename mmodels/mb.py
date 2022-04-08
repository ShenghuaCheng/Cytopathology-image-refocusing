from mmodels.models import build_generator_clear_complex
from mmodels.models import build_generator_clear
from mmodels.models import build_discriminator
from mmodels.models import build_generator_style
from mmodels.models import build_discriminator_cyclegan

from mmodels.srgan import sr_discriminator

from mmodels.multi_scale import build_multi_scale_v3
from mmodels.multi_scale import build_multi_scale_v8
from mmodels.multi_scale import build_generator_clear_dense, build_multi_scale_output


# net choose
d_nets_dict = {0: build_discriminator,
               1: sr_discriminator,
               2: build_discriminator_cyclegan}
g_nets_dict = {0: build_generator_clear,
               2: build_generator_clear_dense,
               3: build_generator_clear_complex,
               6: build_generator_style,
               10: build_multi_scale_v3,
               14: build_multi_scale_v8,
               15: build_multi_scale_output}

pass