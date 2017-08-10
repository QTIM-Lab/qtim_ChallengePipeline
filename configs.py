import config_files.isles_config as isles_config
import config_files.edema_config as edema_config
import config_files.tumor1_config as tumor1_config
import config_files.tumor2_config as tumor2_config
import config_files.nonenhancing1_config as nonenhancing1_config
import config_files.nonenhancing2_config as nonenhancing2_config
import config_files.downsampled_edema_config as downsampled_edema_config
import config_files.upsample_config as upsample_config
import config_files.old_edema_config as old_edema_config
import config_files.fms_config as fms_config

import config_files.edema_config_16 as edema_config_16
import config_files.edema_config_32 as edema_config_32
import config_files.edema_config_64 as edema_config_64
import config_files.edema_config_32_doublefilter as edema_config_32_doublefilter
import config_files.downsampled_edema_config as edema_config_32_downsample
import config_files.edema_config_32_roi90 as edema_config_32_roi90
import config_files.edema_config_32_roi80 as edema_config_32_roi80
import config_files.edema_config_32_roi40 as edema_config_32_roi40



config_map = {'edema16': edema_config_16, 'edema32': edema_config_32, 'edema64': edema_config_64, 'roi90': edema_config_32_roi90, 'roi80': edema_config_32_roi80, 'roi40': edema_config_32_roi40, 'doublefilter': edema_config_32_doublefilter, 'edemadownsample': edema_config_32_downsample}
