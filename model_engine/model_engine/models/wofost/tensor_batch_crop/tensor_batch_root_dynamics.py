"""Class for computing root biomass dynamics and rooting depth

Written by: Will Solow, 2024
"""

from datetime import date

import torch
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    RatesTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)
from model_engine.util import EPS


class WOFOST_Root_Dynamics_TensorBatch(BatchTensorModel):
    """Root biomass dynamics and rooting depth."""

    class Parameters(ParamTemplate):
        RDI = Tensor(-99.0)
        RRI = Tensor(-99.0)
        RDMCR = Tensor(-99.0)
        RDMSOL = Tensor(-99.0)
        TDWI = Tensor(-99.0)
        IAIRDU = Tensor(-99)
        RDRRTB = TensorBatchAfgenTrait()
        RDRROS = TensorBatchAfgenTrait()
        NTHRESH = Tensor(-99.0)
        PTHRESH = Tensor(-99.0)
        KTHRESH = Tensor(-99.0)
        RDRRNPK = TensorBatchAfgenTrait()

    class RateVariables(RatesTemplate):
        RR = Tensor(-99.0)
        GRRT = Tensor(-99.0)
        DRRT1 = Tensor(-99.0)
        DRRT2 = Tensor(-99.0)
        DRRT3 = Tensor(-99.0)
        DRRT = Tensor(-99.0)
        GWRT = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        RD = Tensor(-99.0)
        RDM = Tensor(-99.0)
        WRT = Tensor(-99.0)
        DWRT = Tensor(-99.0)
        TWRT = Tensor(-99.0)

    def __init__(
        self,
        day: date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        p = self.params

        rdmax = torch.max(p.RDI, torch.min(p.RDMCR, p.RDMSOL))
        RDM = rdmax
        RD = p.RDI

        WRT = p.TDWI * self.kiosk.FR
        DWRT = 0.0
        TWRT = WRT + DWRT

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["WRT", "TWRT", "RD"],
            RD=RD,
            RDM=RDM,
            WRT=WRT,
            DWRT=DWRT,
            TWRT=TWRT,
        )

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["GRRT", "DRRT"])

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Calculate state rates for integration"""
        p = self.params
        r = self.rates
        s = self.states
        k = self.kiosk

        r.GRRT = k.FR * k.DMI

        RDRNPK = torch.max(
            torch.max(k.SURFACE_N / p.NTHRESH.clamp(min=EPS), k.SURFACE_P / p.PTHRESH.clamp(min=EPS)),
            k.SURFACE_K / p.KTHRESH.clamp(min=EPS),
        )
        r.DRRT1 = p.RDRRTB(k.DVS)
        r.DRRT2 = p.RDRROS(k.RFOS)
        r.DRRT3 = p.RDRRNPK(RDRNPK)

        r.DRRT = s.WRT * torch.clamp(
            torch.max(r.DRRT1, r.DRRT2 + r.DRRT3),
            torch.tensor([0.0]).to(self.device),
            torch.tensor([1.0]).to(self.device),
        )
        r.GWRT = r.GRRT - r.DRRT

        r.RR = torch.min((s.RDM - s.RD), p.RRI)

        r.RR = torch.where(k.FR == 0.0, 0.0, r.RR)

        r.RR = torch.where(_emerging, 0.0, r.RR)
        r.GRRT = torch.where(_emerging, 0.0, r.GRRT)
        r.DRRT1 = torch.where(_emerging, 0.0, r.DRRT1)
        r.DRRT2 = torch.where(_emerging, 0.0, r.DRRT2)
        r.DRRT3 = torch.where(_emerging, 0.0, r.DRRT3)
        r.DRRT = torch.where(_emerging, 0.0, r.DRRT)
        r.GWRT = torch.where(_emerging, 0.0, r.GWRT)

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate rates for new states"""
        r = self.rates
        s = self.states

        s.WRT = s.WRT + r.GWRT
        s.DWRT = s.DWRT + r.DRRT
        s.TWRT = s.WRT + s.DWRT
        s.RD = s.RD + r.RR

        self.states._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset all states and rates to initial values"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        p = self.params
        s = self.states
        r = self.rates

        rdmax = torch.max(p.RDI, torch.min(p.RDMCR, p.RDMSOL))
        RDM = rdmax
        RD = p.RDI

        WRT = p.TDWI * self.kiosk.FR
        DWRT = 0.0
        TWRT = WRT + DWRT

        s.RD = torch.where(inds, RD, s.RD).detach()
        s.RDM = torch.where(inds, RDM, s.RDM).detach()
        s.WRT = torch.where(inds, WRT, s.WRT).detach()
        s.DWRT = torch.where(inds, DWRT, s.DWRT).detach()
        s.TWRT = torch.where(inds, TWRT, s.TWRT).detach()

        r.RR = torch.where(inds, 0.0, r.RR).detach()
        r.GRRT = torch.where(inds, 0.0, r.GRRT).detach()
        r.DRRT1 = torch.where(inds, 0.0, r.DRRT1).detach()
        r.DRRT2 = torch.where(inds, 0.0, r.DRRT2).detach()
        r.DRRT3 = torch.where(inds, 0.0, r.DRRT3).detach()
        r.DRRT = torch.where(inds, 0.0, r.DRRT).detach()
        r.GWRT = torch.where(inds, 0.0, r.GWRT).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.WRT
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[i, :] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i, :] = getattr(self.rates, v)
            return output_vars

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
