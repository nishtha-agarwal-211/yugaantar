"""Agricultural monsoon vulnerability & heatwave impact module.

Turns a live 10-day ``/simulate`` rollout into CRDI, Kharif crop water stress
and wet-bulb heat-stress indicators on the 32 x 36 contract grid, aggregates
them to agro-climatic districts, estimates affected area and population
exposure, and quantifies how a multiverse shock shifts the risk map.
"""
from .geo import DISTRICTS, DISTRICT_INDEX, LAND, CELL_AREA_KM2
from .indices import assess

__all__ = ["assess", "DISTRICTS", "DISTRICT_INDEX", "LAND", "CELL_AREA_KM2"]
