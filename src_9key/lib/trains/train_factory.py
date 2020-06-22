from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .dddxyz import DddTrainer as DddXyzTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
  'dddxyz': DddXyzTrainer,
}
