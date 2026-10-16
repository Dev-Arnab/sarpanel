"""
sarpanel: analysis-ready Sentinel-1 observations over polygons.

Produces one record per polygon per acquisition, carrying the backscatter
estimate together with the acquisition geometry and speckle statistics needed
to weight it in a statistical model: projected local incidence angle, effective
number of looks, valid pixel count, layover and shadow status, and orbit track.
"""

__version__ = "0.1.0"

from .aoi import AOISet, load_aoi
from .config import RunConfig
from .extract import SceneContext, SceneReader, extract_scene
from .processing import ProcessingConfig

__all__ = [
    "__version__",
    "AOISet",
    "load_aoi",
    "RunConfig",
    "SceneContext",
    "SceneReader",
    "extract_scene",
    "ProcessingConfig",
]
