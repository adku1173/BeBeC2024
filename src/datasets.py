import acoular as ac
import acoupipe.sampler as sp
from acoupipe.datasets.experimental import DatasetMIRACLE as DatasetMIRACLE
from acoupipe.datasets.experimental import DatasetMIRACLEConfig
from acoupipe.datasets.synthetic import DatasetSynthetic as DatasetSynthetic
from acoupipe.datasets.synthetic import DatasetSyntheticConfig
from scipy.stats import norm, randint
from traits.api import Float


class DatasetSyntheticConfigBeBeC2024(DatasetSyntheticConfig):
    """Dataset configuration for the Berlin Beamforming Conference 2024 paper.

    Other than the BaseSynthetic, the following changes are applied:

    * minimum distance between sources of 0.015*aperture
    * uniformly distributed number of sources is used
    * distance to array source plane z=ap
    """

    z_ap = Float(1.0, desc='Distance to array source plane in aperture units')

    def create_nsources_sampler(self):
        return sp.NumericAttributeSampler(
            random_var = randint(self.min_nsources, self.max_nsources+1),
            attribute = 'nsources',
            equal_value = True,
            target=[self.location_sampler],
            )

    def create_grid(self):
        ap = self.mics.aperture
        return ac.RectGrid(y_min=-0.5*ap, y_max=0.5*ap, x_min=-0.5*ap, x_max=0.5*ap,
                                    z=self.z_ap*ap, increment=1/63*ap)

    def create_location_sampler(self):
        ap = self.mics.aperture
        z = self.z_ap*ap
        location_sampler = sp.LocationSampler(
            random_var = (norm(0,0.1688*ap),norm(0,0.1688*ap),norm(z,0)),
            x_bounds = (-0.5*ap,0.5*ap),
            y_bounds = (-0.5*ap,0.5*ap),
            nsources = self.max_nsources,
            mindist = 0.015*ap,
            )
        if self.snap_to_grid:
            location_sampler.grid = self.source_grid
        return location_sampler


    def create_beamformer(self):
        return ac.BeamformerCleansc(
            r_diag=False,
            precision='float32',
            cached=False,
            freq_data = self.freq_data,
            steer = self.steer,
            )



class DatasetMIRACLEConfigBeBeC2024(DatasetMIRACLEConfig):

    def create_nsources_sampler(self):
        return sp.NumericAttributeSampler(
            random_var = randint(self.min_nsources, self.max_nsources+1),
            attribute = 'nsources',
            equal_value = True,
            target=[self.location_sampler],
            )

    def create_beamformer(self):
        return ac.BeamformerCleansc(
            r_diag=False,
            precision='float32',
            cached=False,
            freq_data = self.freq_data,
            steer = self.steer,
            )

