from . import PostProcessingPipeline
from config import Config

p_p_pip = PostProcessingPipeline()
p_p_pip(Config.BAG_FOLDER)

