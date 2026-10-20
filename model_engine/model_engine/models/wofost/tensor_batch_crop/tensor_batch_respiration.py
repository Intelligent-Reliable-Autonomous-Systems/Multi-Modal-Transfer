"""Handles Respiration of crop

Written by: Will Solow, 2025
"""

from datetime import date

import torch
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    RatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)


class WOFOST_Maintenance_Respiration_TensorBatch(BatchTensorModel):
    """Maintenance respiration in WOFOST"""

    class Parameters(ParamTemplate):
        Q10 = Tensor(-99.0)
        RMR = Tensor(-99.0)
        RML = Tensor(-99.0)
        RMS = Tensor(-99.0)
        RMO = Tensor(-99.0)
        RFSETB = TensorBatchAfgenTrait()

    class RateVariables(RatesTemplate):
        PMRES = Tensor(-99.0)

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

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["PMRES"])

    def __call__(self, day: date, drv, _emerging):
        """Calculate the maintenence respiration of the crop"""
        p = self.params
        k = self.kiosk

        RMRES = p.RMR * k.WRT + p.RML * k.WLV + p.RMS * k.WST + p.RMO * k.WSO

        RMRES = RMRES * p.RFSETB(k.DVS)
        TEFF = p.Q10 ** ((drv.TEMP - 25.0) / 10.0)
        self.rates.PMRES = RMRES * TEFF

        self.rates.PMRES = torch.where(_emerging, 0.0, self.rates.PMRES)

        self.rates._update_kiosk()

        return self.rates.PMRES

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""

        self.rates.PMRES = torch.where(inds, 0.0, self.rates.PMRES).detach()

        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.rates.PMRES
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.rates.trait_names():
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
