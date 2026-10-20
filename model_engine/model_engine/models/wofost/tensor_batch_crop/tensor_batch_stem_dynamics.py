"""Handles stem biomass dynamics for crop model

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
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


class WOFOST_Stem_Dynamics_TensorBatch(BatchTensorModel):
    """Implementation of stem biomass dynamics."""

    class Parameters(ParamTemplate):
        RDRSTB = TensorBatchAfgenTrait()
        SSATB = TensorBatchAfgenTrait()
        TDWI = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        WST = Tensor(-99.0)
        DWST = Tensor(-99.0)
        TWST = Tensor(-99.0)
        SAI = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        GRST = Tensor(-99.0)
        DRST = Tensor(-99.0)
        GWST = Tensor(-99.0)

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

        FS = self.kiosk.FS
        FR = self.kiosk.FR
        WST = (self.params.TDWI * (1 - FR)) * FS
        DWST = 0.0
        TWST = WST + DWST

        DVS = self.kiosk.DVS
        SAI = WST * self.params.SSATB(DVS)

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["WST", "TWST"],
            WST=WST,
            DWST=DWST,
            TWST=TWST,
            SAI=SAI,
        )
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["GRST", "DRST"])

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Compute state rates before integration"""
        r = self.rates
        s = self.states
        p = self.params

        r.GRST = self.kiosk.ADMI * self.kiosk.FS
        r.DRST = p.RDRSTB(self.kiosk.DVS) * s.WST
        r.GWST = r.GRST - r.DRST

        r.GRST = torch.where(_emerging, 0.0, r.GRST)
        r.DRST = torch.where(_emerging, 0.0, r.DRST)
        r.GWST = torch.where(_emerging, 0.0, r.GWST)

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate state rates"""
        p = self.params
        r = self.rates
        s = self.states

        s.WST = s.WST + r.GWST
        s.DWST = s.DWST + r.DRST
        s.TWST = s.WST + s.DWST

        s.SAI = s.WST * p.SSATB(self.kiosk.DVS)

        self.states._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        s = self.states
        r = self.rates
        FS = self.kiosk.FS
        FR = self.kiosk.FR
        WST = (self.params.TDWI * (1 - FR)) * FS
        DWST = 0.0
        TWST = WST + DWST

        DVS = self.kiosk.DVS
        SAI = WST * self.params.SSATB(DVS)

        s.WST = torch.where(inds, WST, s.WST).detach()
        s.DWST = torch.where(inds, DWST, s.DWST).detach()
        s.TWST = torch.where(inds, TWST, s.TWST).detach()
        s.SAI = torch.where(inds, SAI, s.SAI).detach()

        r.GRST = torch.where(inds, 0.0, r.GRST).detach()
        r.DRST = torch.where(inds, 0.0, r.DRST).detach()
        r.GWST = torch.where(inds, 0.0, r.GWST).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.WST
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
