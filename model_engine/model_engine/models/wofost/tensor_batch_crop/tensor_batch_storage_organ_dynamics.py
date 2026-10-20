"""Handles storage organ dynamics for crop. Modified from original WOFOST
to include the death of storage organs

Written by Will Solow, 2025
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


class WOFOST_Storage_Organ_Dynamics_TensorBatch(BatchTensorModel):
    """Implementation of storage organ dynamics."""

    class Parameters(ParamTemplate):
        SPA = TensorBatchAfgenTrait()
        TDWI = Tensor(-99.0)
        RDRSOB = TensorBatchAfgenTrait()
        RDRSOF = TensorBatchAfgenTrait()

    class StateVariables(StatesTemplate):
        WSO = Tensor(-99.0)
        DWSO = Tensor(-99.0)
        TWSO = Tensor(-99.0)
        PAI = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        GRSO = Tensor(-99.0)
        DRSO = Tensor(-99.0)
        GWSO = Tensor(-99.0)

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

        FO = self.kiosk.FO
        FR = self.kiosk.FR
        WSO = (p.TDWI * (1 - FR)) * FO
        DWSO = 0.0
        TWSO = WSO + DWSO

        PAI = WSO * p.SPA(self.kiosk.DVS)

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["TWSO", "WSO"],
            WSO=WSO,
            DWSO=DWSO,
            TWSO=TWSO,
            PAI=PAI,
        )

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=[])

        self.zero_tensor = torch.tensor([0.0]).to(self.device)
        self.one_tensor = torch.tensor([1.0]).to(self.device)

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Compute rates for integration"""
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        r.GRSO = k.ADMI * k.FO

        r.DRSO = s.WSO * torch.clamp(p.RDRSOB(k.DVS) + p.RDRSOF(drv.TEMP), self.zero_tensor, self.one_tensor)
        r.GWSO = r.GRSO - r.DRSO

        r.GRSO = torch.where(_emerging, 0.0, r.GRSO)
        r.DRSO = torch.where(_emerging, 0.0, r.DRSO)
        r.GWSO = torch.where(_emerging, 0.0, r.GWSO)

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate rates"""
        p = self.params
        r = self.rates
        s = self.states

        s.WSO = s.WSO + r.GWSO
        s.DWSO = s.DWSO + r.DRSO
        s.TWSO = s.WSO + s.DWSO
        s.PAI = s.WSO * p.SPA(self.kiosk.DVS)

        self.states._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        p = self.params
        s = self.states
        r = self.rates

        FO = self.kiosk.FO
        FR = self.kiosk.FR
        WSO = (p.TDWI * (1 - FR)) * FO
        DWSO = 0.0
        TWSO = WSO + DWSO

        PAI = WSO * p.SPA(self.kiosk.DVS)

        s.WSO = torch.where(inds, WSO, s.WSO).detach()
        s.DWSO = torch.where(inds, DWSO, s.DWSO).detach()
        s.TWSO = torch.where(inds, TWSO, s.TWSO).detach()
        s.PAI = torch.where(inds, PAI, s.PAI).detach()

        r.GRSO = torch.where(inds, 0.0, r.GRSO).detach()
        r.DRSO = torch.where(inds, 0.0, r.DRSO).detach()
        r.GWSO = torch.where(inds, 0.0, r.GWSO).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.WSO
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
